import { randomUUID } from 'crypto'
import {
    Batch,
    ClusterBatch,
    ClusterScanCursor,
    GlideClient,
    GlideClusterClient,
    GlideFt,
    Field,
    GlideClientConfiguration
} from '@valkey/valkey-glide'
import { VectorStore } from '@langchain/core/vectorstores'
import type { EmbeddingsInterface } from '@langchain/core/embeddings'
import { Document } from '@langchain/core/documents'
import { ICommonObject, INode, INodeData, INodeOutputsValue, INodeParams, IndexingResult } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { index } from '../../../src/indexing'

// ValkeyVectorStore — adapted from @langchain/valkey (langchainjs)
// https://github.com/daric93/langchainjs/tree/feature/valkey-vector-store

export enum VectorAlgorithms {
    FLAT = 'FLAT',
    HNSW = 'HNSW'
}

export type CreateSchemaVectorField<T extends VectorAlgorithms, A extends Record<string, unknown>> = {
    ALGORITHM: T
    DISTANCE_METRIC: 'L2' | 'IP' | 'COSINE'
    INITIAL_CAP?: number
} & A

export type CreateSchemaFlatVectorField = CreateSchemaVectorField<VectorAlgorithms.FLAT, { BLOCK_SIZE?: number }>

export type CreateSchemaHNSWVectorField = CreateSchemaVectorField<
    VectorAlgorithms.HNSW,
    { M?: number; EF_CONSTRUCTION?: number; EF_RUNTIME?: number }
>

export interface ValkeyVectorStoreConfig {
    valkeyClient: GlideClient | GlideClusterClient
    indexName: string
    indexOptions?: CreateSchemaFlatVectorField | CreateSchemaHNSWVectorField
    createIndexOptions?: Record<string, unknown>
    keyPrefix?: string
    contentKey?: string
    metadataKey?: string
    vectorKey?: string
    filter?: ValkeyVectorStoreFilterType
    ttl?: number
}

export interface ValkeyAddOptions {
    keys?: string[]
    batchSize?: number
}

export type ValkeyVectorStoreFilterType = string[] | string

export class ValkeyVectorStore extends VectorStore {
    declare FilterType: ValkeyVectorStoreFilterType

    valkeyClient: GlideClient | GlideClusterClient

    indexName: string

    indexOptions: CreateSchemaFlatVectorField | CreateSchemaHNSWVectorField

    createIndexOptions: Record<string, unknown>

    keyPrefix: string

    contentKey: string

    metadataKey: string

    vectorKey: string

    filter?: ValkeyVectorStoreFilterType

    ttl?: number

    _vectorstoreType(): string {
        return 'valkey'
    }

    constructor(embeddings: EmbeddingsInterface, _dbConfig: ValkeyVectorStoreConfig) {
        super(embeddings, _dbConfig)
        this.valkeyClient = _dbConfig.valkeyClient
        this.indexName = _dbConfig.indexName
        this.indexOptions = _dbConfig.indexOptions ?? {
            ALGORITHM: VectorAlgorithms.HNSW,
            DISTANCE_METRIC: 'COSINE'
        }
        this.keyPrefix = _dbConfig.keyPrefix ?? `doc:${this.indexName}:`
        this.contentKey = _dbConfig.contentKey ?? 'content'
        this.metadataKey = _dbConfig.metadataKey ?? 'metadata'
        this.vectorKey = _dbConfig.vectorKey ?? 'content_vector'
        this.filter = _dbConfig.filter
        if (_dbConfig.ttl !== undefined && _dbConfig.ttl <= 0) {
            throw new Error(`TTL must be a positive integer, got ${_dbConfig.ttl}`)
        }
        this.ttl = _dbConfig.ttl
        this.createIndexOptions = {
            ON: 'HASH',
            PREFIX: this.keyPrefix,
            ...(_dbConfig.createIndexOptions || {})
        }
    }

    async checkIndexExists(): Promise<boolean> {
        try {
            await GlideFt.info(this.valkeyClient, this.indexName)
            return true
        } catch (err) {
            if ((err as Error)?.message.includes('unknown command')) {
                throw new Error(
                    'Failed to run FT.INFO command. Please ensure that your Valkey instance has the valkey-search module enabled.'
                )
            }
            return false
        }
    }

    async createIndex(dimensions = 1536): Promise<void> {
        if (await this.checkIndexExists()) {
            return
        }

        // TAG field requires Valkey Search >= 1.2 (Valkey >= 9.1)
        const schema: Field[] = [
            {
                type: 'TAG',
                name: `${this.metadataKey}_tags`,
                separator: ','
            },
            {
                type: 'VECTOR',
                name: this.vectorKey,
                attributes: {
                    algorithm: this.indexOptions.ALGORITHM,
                    type: 'FLOAT32',
                    dimensions: dimensions,
                    distanceMetric: this.indexOptions.DISTANCE_METRIC
                }
            }
        ]

        await GlideFt.create(this.valkeyClient, this.indexName, schema, {
            dataType: 'HASH',
            prefixes: [this.keyPrefix]
        }).catch((err) => {
            const msg = (err as Error)?.message || ''
            if (msg.includes('already exists')) {
                return // Index was created concurrently — safe to continue
            }
            if (msg.includes('TAG') || msg.includes('unsupported') || msg.includes('unknown field type')) {
                throw new Error(
                    'Failed to create index with TAG field. Metadata filtering requires Valkey Search >= 1.2 (Valkey >= 9.1). ' +
                        `Original error: ${msg}`
                )
            }
            throw err
        })
    }

    async dropIndex(): Promise<boolean> {
        try {
            await GlideFt.dropindex(this.valkeyClient, this.indexName)
            return true
        } catch {
            return false
        }
    }

    async delete(params: { deleteAll: boolean } | { ids: string[] }): Promise<void> {
        if ('ids' in params && params.ids?.length === 0) return
        if ('deleteAll' in params && params.deleteAll) {
            await this.dropIndex()
            // Scan and delete orphaned HASH keys
            if (this.valkeyClient instanceof GlideClusterClient) {
                let cursor = new ClusterScanCursor()
                do {
                    const [next, keys] = await this.valkeyClient.scan(cursor, { match: `${this.keyPrefix}*`, count: 100 })
                    cursor = next
                    if (keys.length) await this.valkeyClient.del(keys as string[])
                } while (!cursor.isFinished())
            } else {
                let cursor = '0'
                do {
                    const result = await (this.valkeyClient as GlideClient).scan(cursor, {
                        match: `${this.keyPrefix}*`,
                        count: 100
                    })
                    cursor = String(result[0])
                    const keys = result[1] as string[]
                    if (keys.length) await this.valkeyClient.del(keys)
                } while (cursor !== '0')
            }
        } else if ('ids' in params && params.ids && params.ids.length > 0) {
            const keys = params.ids.map((id) => (id.startsWith(this.keyPrefix) ? id : `${this.keyPrefix}${id}`))
            await this.valkeyClient.del(keys)
        } else {
            throw new Error(`Invalid parameters passed to "delete".`)
        }
    }

    async addDocuments(documents: Document[], options?: ValkeyAddOptions) {
        const texts = documents.map(({ pageContent }) => pageContent)
        return this.addVectors(await this.embeddings.embedDocuments(texts), documents, options)
    }

    async addVectors(vectors: number[][], documents: Document[], { keys, batchSize = 100 }: ValkeyAddOptions = {}) {
        if (!vectors.length || !vectors[0].length) {
            throw new Error('No vectors provided')
        }
        if (vectors.length !== documents.length) {
            throw new Error(`Vectors length (${vectors.length}) must match documents length (${documents.length})`)
        }
        await this.createIndex(vectors[0].length)

        const commands: Array<{ key: string; fields: Record<string, string | Buffer> }> = []

        for (let idx = 0; idx < vectors.length; idx += 1) {
            const vector = vectors[idx]
            const key = keys && keys.length ? keys[idx] : `${this.keyPrefix}${randomUUID()}`
            const metadata = documents[idx] && documents[idx].metadata ? documents[idx].metadata : {}

            const hashFields: Record<string, string | Buffer> = {
                [this.vectorKey]: this.getFloat32Buffer(vector),
                [this.contentKey]: documents[idx]?.pageContent || '',
                [this.metadataKey]: JSON.stringify(metadata),
                [`${this.metadataKey}_tags`]: Object.values(metadata)
                    .flatMap((v) =>
                        Array.isArray(v)
                            ? v.filter((x) => x != null && typeof x !== 'object').map(String)
                            : v != null && typeof v !== 'object'
                            ? [String(v)]
                            : []
                    )
                    .join(',')
            }

            commands.push({ key, fields: hashFields })

            if (commands.length >= batchSize || idx === vectors.length - 1) {
                if (this.valkeyClient instanceof GlideClusterClient) {
                    const batch = new ClusterBatch(false) // non-atomic: allows multi-slot routing
                    for (const { key, fields } of commands) {
                        batch.hset(key, fields)
                        if (this.ttl) {
                            batch.expire(key, this.ttl)
                        }
                    }
                    await this.valkeyClient.exec(batch, true, { retryStrategy: { retryServerError: true, retryConnectionError: true } })
                } else {
                    const batch = new Batch(false)
                    for (const { key, fields } of commands) {
                        batch.hset(key, fields)
                        if (this.ttl) {
                            batch.expire(key, this.ttl)
                        }
                    }
                    await (this.valkeyClient as GlideClient).exec(batch, true)
                }
                commands.length = 0
            }
        }
    }

    async similaritySearchVectorWithScore(query: number[], k: number, filter?: ValkeyVectorStoreFilterType): Promise<[Document, number][]> {
        if (filter && this.filter) {
            throw new Error('cannot provide both `filter` and `this.filter`')
        }

        if (!(await this.checkIndexExists())) {
            return []
        }

        const _filter = filter ?? this.filter
        const [queryStr, options] = this.buildQuery(query, k, _filter)

        const searchOptions = {
            params: [{ key: 'vector', value: options.PARAMS.vector }],
            returnFields: options.RETURN.map((field) => ({ fieldIdentifier: field })),
            sortby: options.SORTBY,
            dialect: options.DIALECT,
            limit: { offset: options.LIMIT.from, count: options.LIMIT.size }
        }

        const results = await GlideFt.search(this.valkeyClient, this.indexName, queryStr, searchOptions)
        return this.parseSearchResults(results)
    }

    private parseSearchResults(results: unknown): [Document, number][] {
        const result: [Document, number][] = []

        if (Array.isArray(results) && results.length > 1) {
            const documents = results[1]
            if (Array.isArray(documents)) {
                for (const doc of documents) {
                    if (Array.isArray(doc?.value)) {
                        const fieldsObj: Record<string, unknown> = {}
                        for (const field of doc.value) {
                            if (field?.key !== undefined && field?.value !== undefined) {
                                fieldsObj[String(field.key)] = field.value
                            }
                        }

                        if (fieldsObj.vector_score !== undefined) {
                            let metadata: Record<string, unknown> = {}
                            try {
                                metadata = JSON.parse((fieldsObj[this.metadataKey] ?? '{}') as string)
                            } catch {
                                metadata = {}
                            }

                            result.push([
                                new Document({
                                    pageContent: (fieldsObj[this.contentKey] ?? '') as string,
                                    metadata
                                }),
                                Number(fieldsObj.vector_score)
                            ])
                        }
                    }
                }
            }
        }

        return result
    }

    private buildQuery(
        query: number[],
        k: number,
        _filter?: ValkeyVectorStoreFilterType
    ): [string, { PARAMS: { vector: Buffer }; RETURN: string[]; SORTBY: string; DIALECT: number; LIMIT: { from: number; size: number } }] {
        const vectorScoreField = 'vector_score'
        let hybridFields = '*'

        if (_filter) {
            const tags = Array.isArray(_filter) ? _filter : [_filter]
            if (tags.length > 0) {
                const escaped = tags.map((t) => t.replace(/\\/g, '\\\\').replace(/[,.<>{}[\]"':;!@#$%^&*()\-+=~|/ ]/g, '\\$&'))
                hybridFields = `@${this.metadataKey}_tags:{${escaped.join(' | ')}}`
            }
        }

        const baseQuery = `${hybridFields}=>[KNN ${k} @${this.vectorKey} $vector AS ${vectorScoreField}]`
        const returnFields = [this.metadataKey, this.contentKey, vectorScoreField]

        const options = {
            PARAMS: { vector: this.getFloat32Buffer(query) },
            RETURN: returnFields,
            SORTBY: vectorScoreField,
            DIALECT: 2,
            LIMIT: { from: 0, size: k }
        }

        return [baseQuery, options]
    }

    private getFloat32Buffer(vector: number[]): Buffer {
        return Buffer.from(new Float32Array(vector).buffer)
    }

    static async fromDocuments(
        docs: Document[],
        embeddings: EmbeddingsInterface,
        dbConfig: ValkeyVectorStoreConfig,
        docsOptions?: ValkeyAddOptions
    ): Promise<ValkeyVectorStore> {
        const instance = new this(embeddings, dbConfig)
        await instance.addDocuments(docs, docsOptions)
        return instance
    }

    static fromTexts(
        texts: string[],
        metadatas: object[] | object,
        embeddings: EmbeddingsInterface,
        dbConfig: ValkeyVectorStoreConfig,
        docsOptions?: ValkeyAddOptions
    ): Promise<ValkeyVectorStore> {
        const docs: Document[] = []
        for (let i = 0; i < texts.length; i += 1) {
            const metadata = Array.isArray(metadatas) ? metadatas[i] : metadatas
            docs.push(new Document({ pageContent: texts[i], metadata }))
        }
        return ValkeyVectorStore.fromDocuments(docs, embeddings, dbConfig, docsOptions)
    }
}

// Flowise Node Wrapper

interface ValkeyConnectionConfig {
    host: string
    port: number
    username?: string
    password?: string
    useTLS?: boolean
}

function parseConnectionUrl(url: string): ValkeyConnectionConfig {
    const parsed = new URL(url)
    return {
        host: parsed.hostname || '127.0.0.1',
        port: parseInt(parsed.port, 10) || 6379,
        username: parsed.username ? decodeURIComponent(parsed.username) : undefined,
        password: parsed.password ? decodeURIComponent(parsed.password) : undefined,
        useTLS: parsed.protocol === 'rediss:' || parsed.protocol === 'valkeys:'
    }
}

async function createGlideClient(config: ValkeyConnectionConfig): Promise<GlideClient> {
    const clientConfig: GlideClientConfiguration = {
        addresses: [{ host: config.host, port: config.port }],
        requestTimeout: 5000,
        ...(config.username || config.password
            ? {
                  credentials: {
                      ...(config.username && { username: config.username }),
                      password: config.password ?? ''
                  }
              }
            : {}),
        ...(config.useTLS ? { useTLS: true } : {})
    }
    return await GlideClient.createClient(clientConfig)
}

function getConnectionConfig(credentialData: ICommonObject, nodeData: INodeData): ValkeyConnectionConfig {
    const url = getCredentialParam('valkeyUrl', credentialData, nodeData)
    if (url && url !== '') {
        return parseConnectionUrl(url)
    }
    const host = getCredentialParam('valkeyHost', credentialData, nodeData) || '127.0.0.1'
    const port = parseInt(getCredentialParam('valkeyPort', credentialData, nodeData) || '6379', 10)
    const username = getCredentialParam('valkeyUser', credentialData, nodeData) || undefined
    const password = getCredentialParam('valkeyPassword', credentialData, nodeData) || undefined
    const useTLS = getCredentialParam('valkeyTls', credentialData, nodeData) === 'true'
    return { host, port, username, password, useTLS }
}

class Valkey_VectorStores implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    badge: string
    baseClasses: string[]
    inputs: INodeParams[]
    credential: INodeParams
    outputs: INodeOutputsValue[]

    constructor() {
        this.label = 'Valkey'
        this.name = 'valkey'
        this.version = 1.0
        this.description =
            'Upsert embedded data and perform similarity search upon query using Valkey, a high-performance open-source key-value store. Requires Valkey Search >= 1.2 (Valkey >= 9.1) for metadata filtering.'
        this.type = 'Valkey'
        this.icon = 'valkey.svg'
        this.category = 'Vector Stores'
        this.badge = 'NEW'
        this.baseClasses = [this.type, 'VectorStoreRetriever', 'BaseRetriever']
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['valkeyUrlApi', 'valkeyApi']
        }
        this.inputs = [
            {
                label: 'Document',
                name: 'document',
                type: 'Document',
                list: true,
                optional: true
            },
            {
                label: 'Embeddings',
                name: 'embeddings',
                type: 'Embeddings'
            },
            {
                label: 'Record Manager',
                name: 'recordManager',
                type: 'RecordManager',
                description: 'Keep track of the record to prevent duplication',
                optional: true
            },
            {
                label: 'Index Name',
                name: 'indexName',
                description:
                    'Name of the Valkey search index used for vector similarity queries. Will be created automatically on upsert if it does not exist.',
                placeholder: '<VECTOR_INDEX_NAME>',
                type: 'string'
            },
            {
                label: 'Replace Index on Upsert',
                name: 'replaceIndex',
                description: 'Selecting this option will delete the existing index and recreate a new one when upserting',
                default: false,
                type: 'boolean'
            },
            {
                label: 'Content Field',
                name: 'contentKey',
                description: 'Name of the field (column) that contains the actual content',
                type: 'string',
                default: 'content',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Metadata Field',
                name: 'metadataKey',
                description: 'Name of the field (column) that contains the metadata of the document',
                type: 'string',
                default: 'metadata',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Vector Field',
                name: 'vectorKey',
                description: 'Name of the field (column) that contains the vector',
                type: 'string',
                default: 'content_vector',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Valkey Metadata Filter',
                name: 'valkeyMetadataFilter',
                type: 'json',
                description:
                    'Filter documents by metadata tags. Provide a JSON array of strings to match against the metadata field using OR logic, e.g. ["tag1", "tag2"]',
                optional: true,
                additionalParams: true,
                acceptVariable: true
            },
            {
                label: 'Top K',
                name: 'topK',
                description: 'Number of top results to fetch. Default to 4',
                placeholder: '4',
                type: 'number',
                additionalParams: true,
                optional: true
            }
        ]
        this.outputs = [
            {
                label: 'Valkey Retriever',
                name: 'retriever',
                baseClasses: this.baseClasses
            },
            {
                label: 'Valkey Vector Store',
                name: 'vectorStore',
                baseClasses: [this.type, ...getBaseClasses(ValkeyVectorStore)]
            }
        ]
    }

    // @ts-expect-error — upsert returns Partial<IndexingResult>; INode expects full IndexingResult | void
    vectorStoreMethods = {
        async upsert(nodeData: INodeData, options: ICommonObject): Promise<Partial<IndexingResult>> {
            const credentialData = await getCredentialData(nodeData.credential ?? '', options)
            const connectionConfig = getConnectionConfig(credentialData, nodeData)
            const indexName = nodeData.inputs?.indexName as string
            const embeddings = nodeData.inputs?.embeddings as EmbeddingsInterface
            const replaceIndex = nodeData.inputs?.replaceIndex as boolean
            const contentKey = (nodeData.inputs?.contentKey as string) || 'content'
            const metadataKey = (nodeData.inputs?.metadataKey as string) || 'metadata'
            const vectorKey = (nodeData.inputs?.vectorKey as string) || 'content_vector'
            const recordManager = nodeData.inputs?.recordManager

            const docs = nodeData.inputs?.document as Document[]
            const flattenDocs = docs && docs.length ? docs.flat() : []
            const finalDocs: Document[] = []
            for (let i = 0; i < flattenDocs.length; i += 1) {
                if (flattenDocs[i] && flattenDocs[i].pageContent) {
                    finalDocs.push(new Document(flattenDocs[i]))
                }
            }

            const client = await createGlideClient(connectionConfig)
            try {
                const vectorStore = new ValkeyVectorStore(embeddings, {
                    valkeyClient: client,
                    indexName,
                    contentKey,
                    metadataKey,
                    vectorKey
                })

                if (replaceIndex) {
                    await vectorStore.dropIndex()
                }

                if (recordManager) {
                    await recordManager.createSchema()
                    const res = await index({
                        docsSource: finalDocs,
                        recordManager,
                        vectorStore,
                        options: {
                            cleanup: recordManager?.cleanup,
                            sourceIdKey: recordManager?.sourceIdKey ?? 'source',
                            vectorStoreName: indexName
                        }
                    })
                    return res
                } else {
                    await vectorStore.addDocuments(finalDocs)
                    return { numAdded: finalDocs.length, addedDocs: finalDocs }
                }
            } catch (e) {
                throw e instanceof Error ? e : new Error(String(e))
            } finally {
                client.close()
            }
        },
        async delete(nodeData: INodeData, ids: string[], options: ICommonObject): Promise<void> {
            const credentialData = await getCredentialData(nodeData.credential ?? '', options)
            const connectionConfig = getConnectionConfig(credentialData, nodeData)
            const indexName = nodeData.inputs?.indexName as string
            const embeddings = nodeData.inputs?.embeddings as EmbeddingsInterface
            const contentKey = (nodeData.inputs?.contentKey as string) || 'content'
            const metadataKey = (nodeData.inputs?.metadataKey as string) || 'metadata'
            const vectorKey = (nodeData.inputs?.vectorKey as string) || 'content_vector'
            const recordManager = nodeData.inputs?.recordManager

            const client = await createGlideClient(connectionConfig)
            try {
                const vectorStore = new ValkeyVectorStore(embeddings, {
                    valkeyClient: client,
                    indexName,
                    contentKey,
                    metadataKey,
                    vectorKey
                })

                if (recordManager) {
                    const vectorStoreName = indexName
                    await recordManager.createSchema()
                    ;(recordManager as any).namespace = (recordManager as any).namespace + '_' + vectorStoreName
                    const keys: string[] = await recordManager.listKeys({})
                    await vectorStore.delete({ ids: keys })
                    await recordManager.deleteKeys(keys)
                } else {
                    await vectorStore.delete({ ids })
                }
            } catch (e) {
                throw e instanceof Error ? e : new Error(String(e))
            } finally {
                client.close()
            }
        }
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const connectionConfig = getConnectionConfig(credentialData, nodeData)
        const indexName = nodeData.inputs?.indexName as string
        const embeddings = nodeData.inputs?.embeddings as EmbeddingsInterface
        const topK = nodeData.inputs?.topK as string
        const k = topK ? Math.max(1, parseInt(topK, 10)) : 4
        const output = nodeData.outputs?.output as string
        const contentKey = (nodeData.inputs?.contentKey as string) || 'content'
        const metadataKey = (nodeData.inputs?.metadataKey as string) || 'metadata'
        const vectorKey = (nodeData.inputs?.vectorKey as string) || 'content_vector'
        const valkeyMetadataFilter = nodeData.inputs?.valkeyMetadataFilter

        let filter: ValkeyVectorStoreFilterType | undefined
        if (valkeyMetadataFilter) {
            const parsed = typeof valkeyMetadataFilter === 'object' ? valkeyMetadataFilter : JSON.parse(valkeyMetadataFilter)
            if (!Array.isArray(parsed) && typeof parsed !== 'string') {
                throw new Error('valkeyMetadataFilter must be a JSON array of strings or a single string')
            }
            filter = parsed
        }

        // Persistent client for the vector store's lifetime. Matches Postgres/TypeORM pattern in
        // Flowise — no framework teardown hook exists. Recreating per query would forfeit GLIDE's
        // multiplexed connection and automatic reconnection. One TCP connection per chatflow instance.
        const client = await createGlideClient(connectionConfig)

        const vectorStore = new ValkeyVectorStore(embeddings, {
            valkeyClient: client,
            indexName,
            contentKey,
            metadataKey,
            vectorKey,
            filter
        })

        if (output === 'retriever') {
            return vectorStore.asRetriever(k)
        } else if (output === 'vectorStore') {
            ;(vectorStore as any).k = k
            return vectorStore
        }
        return vectorStore
    }
}

module.exports = { nodeClass: Valkey_VectorStores, ValkeyVectorStore, parseConnectionUrl, createGlideClient, getConnectionConfig }
