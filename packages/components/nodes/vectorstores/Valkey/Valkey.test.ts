import { Document } from '@langchain/core/documents'

// Mock @valkey/valkey-glide
const mockExec = jest.fn().mockResolvedValue([])
const mockHset = jest.fn().mockReturnThis()
const mockExpire = jest.fn().mockReturnThis()
const mockDel = jest.fn().mockResolvedValue(1)
const mockScan = jest.fn().mockResolvedValue(['0', []])
const mockClose = jest.fn()
const mockCustomCommand = jest.fn().mockResolvedValue([])

const mockGlideClient = {
    exec: mockExec,
    del: mockDel,
    scan: mockScan,
    close: mockClose,
    customCommand: mockCustomCommand
}

jest.mock('@valkey/valkey-glide', () => ({
    GlideClient: {
        createClient: jest.fn().mockResolvedValue(mockGlideClient)
    },
    GlideClusterClient: class GlideClusterClient {},
    ClusterScanCursor: class ClusterScanCursor {
        _finished = true
        constructor(finished = true) {
            this._finished = finished
        }
        isFinished() {
            return this._finished
        }
    },
    GlideFt: {
        search: jest.fn().mockResolvedValue([0, []]),
        info: jest.fn().mockResolvedValue({ numDocs: '0' }),
        create: jest.fn().mockResolvedValue('OK'),
        dropindex: jest.fn().mockResolvedValue('OK')
    },
    Batch: jest.fn().mockImplementation(() => ({
        hset: mockHset,
        expire: mockExpire
    })),
    ClusterBatch: jest.fn().mockImplementation(() => ({
        hset: mockHset,
        expire: mockExpire
    })),
    Field: {}
}))

const mockGetCredentialData = jest.fn().mockResolvedValue({})
const mockGetCredentialParam = jest.fn()

jest.mock('../../../src/utils', () => ({
    getBaseClasses: jest.fn(() => ['VectorStore']),
    getCredentialData: (...args: any[]) => mockGetCredentialData(...args),
    getCredentialParam: (...args: any[]) => mockGetCredentialParam(...args)
}))

jest.mock('../../../src/indexing', () => ({
    index: jest.fn().mockResolvedValue({ numAdded: 1, addedDocs: [] })
}))

const { ValkeyVectorStore, parseConnectionUrl, nodeClass: Valkey_VectorStores, getConnectionConfig } = require('./Valkey')

class FakeEmbeddings {
    async embedDocuments(texts: string[]): Promise<number[][]> {
        return texts.map(() => new Array(128).fill(0.1))
    }
    async embedQuery(): Promise<number[]> {
        return new Array(128).fill(0.1)
    }
}

describe('parseConnectionUrl', () => {
    it('should parse basic valkey URL', () => {
        const config = parseConnectionUrl('valkey://localhost:6379')
        expect(config).toEqual({
            host: 'localhost',
            port: 6379,
            username: undefined,
            password: undefined,
            useTLS: false
        })
    })

    it('should parse URL with credentials', () => {
        const config = parseConnectionUrl('valkey://user:pass@myhost:6380')
        expect(config.host).toBe('myhost')
        expect(config.port).toBe(6380)
        expect(config.username).toBe('user')
        expect(config.password).toBe('pass')
    })

    it('should decode URL-encoded credentials', () => {
        const config = parseConnectionUrl('valkey://user%40name:p%40ss%23word@host:6379')
        expect(config.username).toBe('user@name')
        expect(config.password).toBe('p@ss#word')
    })

    it('should detect TLS from valkeys:// protocol', () => {
        const config = parseConnectionUrl('valkeys://host:6380')
        expect(config.useTLS).toBe(true)
    })

    it('should detect TLS from rediss:// protocol', () => {
        const config = parseConnectionUrl('rediss://host:6380')
        expect(config.useTLS).toBe(true)
    })

    it('should not enable TLS for redis:// protocol', () => {
        const config = parseConnectionUrl('redis://host:6379')
        expect(config.useTLS).toBe(false)
    })

    it('should default port to 6379 when not specified', () => {
        const config = parseConnectionUrl('valkey://host')
        expect(config.port).toBe(6379)
    })

    it('should parse password-only auth (no username)', () => {
        const config = parseConnectionUrl('valkey://:secret@localhost:6379')
        expect(config.host).toBe('localhost')
        expect(config.password).toBe('secret')
        expect(config.username).toBeUndefined()
    })

    it('should handle password with percent-encoded special chars', () => {
        const config = parseConnectionUrl('valkey://:%25percent%26amp@host:6379')
        expect(config.password).toBe('%percent&amp')
    })

    it('should throw on invalid URL', () => {
        expect(() => parseConnectionUrl('')).toThrow()
        expect(() => parseConnectionUrl('not-a-url')).toThrow()
    })

    it('should default host to 127.0.0.1 when hostname is empty', () => {
        // new URL('valkey://') parses with empty hostname
        const config = parseConnectionUrl('valkey://')
        expect(config.host).toBe('127.0.0.1')
    })
})

describe('ValkeyVectorStore', () => {
    let store: any

    beforeEach(() => {
        jest.clearAllMocks()
        store = new ValkeyVectorStore(new FakeEmbeddings(), {
            valkeyClient: mockGlideClient,
            indexName: 'test-idx',
            keyPrefix: 'doc:test:'
        })
    })

    describe('buildQuery', () => {
        it('should produce wildcard query without filter', () => {
            const [query] = store.buildQuery([0.1], 5, undefined)
            expect(query).toBe('*=>[KNN 5 @content_vector $vector AS vector_score]')
        })

        it('should produce filter expression from string filter', () => {
            const [query] = store.buildQuery([0.1], 3, 'mytag')
            expect(query).toBe('@metadata_tags:{mytag}=>[KNN 3 @content_vector $vector AS vector_score]')
        })

        it('should produce OR expression from array filter', () => {
            const [query] = store.buildQuery([0.1], 2, ['tag1', 'tag2'])
            expect(query).toBe('@metadata_tags:{tag1 | tag2}=>[KNN 2 @content_vector $vector AS vector_score]')
        })

        it.each([
            ['tag:with,specials', 'tag\\:with\\,specials'],
            ['tag{1}[2]', 'tag\\{1\\}\\[2\\]'],
            ['hello world', 'hello\\ world'],
            ["it's@here", "it\\'s\\@here"]
        ])('should escape special characters in filter tag "%s"', (input, expected) => {
            const [query] = store.buildQuery([0.1], 1, [input])
            expect(query).toContain(expected)
        })

        it('should fall back to wildcard query for empty filter array', () => {
            const [query] = store.buildQuery([0.1], 5, [])
            expect(query).toBe('*=>[KNN 5 @content_vector $vector AS vector_score]')
        })

        it('should return correct options structure', () => {
            const [, options] = store.buildQuery([0.1, 0.2], 4, undefined)
            expect(options.RETURN).toEqual(['metadata', 'content', 'vector_score'])
            expect(options.SORTBY).toBe('vector_score')
            expect(options.DIALECT).toBe(2)
            expect(options.LIMIT).toEqual({ from: 0, size: 4 })
            expect(options.PARAMS.vector).toBeInstanceOf(Buffer)
        })

        it('should encode vector as Float32 buffer', () => {
            const [, options] = store.buildQuery([1.0, 2.0], 1, undefined)
            const buf = options.PARAMS.vector
            const floats = new Float32Array(buf.buffer, buf.byteOffset, 2)
            expect(floats[0]).toBeCloseTo(1.0)
            expect(floats[1]).toBeCloseTo(2.0)
        })
    })

    describe('parseSearchResults', () => {
        it('should return empty array for empty results', () => {
            expect(store.parseSearchResults([0, []])).toEqual([])
        })

        it('should return empty array for non-array input', () => {
            expect(store.parseSearchResults(null)).toEqual([])
            expect(store.parseSearchResults(undefined)).toEqual([])
            expect(store.parseSearchResults('string')).toEqual([])
        })

        it('should parse valid search results', () => {
            const raw = [
                1,
                [
                    {
                        value: [
                            { key: 'content', value: 'hello world' },
                            { key: 'metadata', value: '{"source":"test"}' },
                            { key: 'vector_score', value: '0.5' }
                        ]
                    }
                ]
            ]
            const results = store.parseSearchResults(raw)
            expect(results.length).toBe(1)
            expect(results[0][0]).toBeInstanceOf(Document)
            expect(results[0][0].pageContent).toBe('hello world')
            expect(results[0][0].metadata).toEqual({ source: 'test' })
            expect(results[0][1]).toBe(0.5)
        })

        it('should parse multiple documents correctly', () => {
            const raw = [
                3,
                [
                    {
                        value: [
                            { key: 'content', value: 'first' },
                            { key: 'metadata', value: '{"idx":1}' },
                            { key: 'vector_score', value: '0.1' }
                        ]
                    },
                    {
                        value: [
                            { key: 'content', value: 'second' },
                            { key: 'metadata', value: '{"idx":2}' },
                            { key: 'vector_score', value: '0.3' }
                        ]
                    },
                    {
                        value: [
                            { key: 'content', value: 'third' },
                            { key: 'metadata', value: '{"idx":3}' },
                            { key: 'vector_score', value: '0.7' }
                        ]
                    }
                ]
            ]
            const results = store.parseSearchResults(raw)
            expect(results.length).toBe(3)
            expect(results[0][0].pageContent).toBe('first')
            expect(results[1][0].pageContent).toBe('second')
            expect(results[2][0].pageContent).toBe('third')
            expect(results[0][1]).toBe(0.1)
            expect(results[2][1]).toBe(0.7)
        })

        it('should default to empty string when content field is missing', () => {
            const raw = [
                1,
                [
                    {
                        value: [
                            { key: 'metadata', value: '{"a":1}' },
                            { key: 'vector_score', value: '0.2' }
                        ]
                    }
                ]
            ]
            const results = store.parseSearchResults(raw)
            expect(results.length).toBe(1)
            expect(results[0][0].pageContent).toBe('')
            expect(results[0][0].metadata).toEqual({ a: 1 })
        })

        it('should handle malformed metadata JSON gracefully', () => {
            const raw = [
                1,
                [
                    {
                        value: [
                            { key: 'content', value: 'doc' },
                            { key: 'metadata', value: 'not-json{' },
                            { key: 'vector_score', value: '0.1' }
                        ]
                    }
                ]
            ]
            const results = store.parseSearchResults(raw)
            expect(results.length).toBe(1)
            expect(results[0][0].metadata).toEqual({})
        })

        it('should skip entries without vector_score', () => {
            const raw = [
                1,
                [
                    {
                        value: [
                            { key: 'content', value: 'doc' },
                            { key: 'metadata', value: '{}' }
                        ]
                    }
                ]
            ]
            const results = store.parseSearchResults(raw)
            expect(results.length).toBe(0)
        })
    })

    describe('delete', () => {
        it('should no-op for empty ids array', async () => {
            await store.delete({ ids: [] })
            expect(mockDel).not.toHaveBeenCalled()
        })

        it('should delete keys with prefix prepended when missing', async () => {
            await store.delete({ ids: ['abc', 'def'] })
            expect(mockDel).toHaveBeenCalledWith(['doc:test:abc', 'doc:test:def'])
        })

        it('should not double-prefix keys that already have prefix', async () => {
            await store.delete({ ids: ['doc:test:abc'] })
            expect(mockDel).toHaveBeenCalledWith(['doc:test:abc'])
        })

        it('should throw for invalid params', async () => {
            await expect(store.delete({ invalid: true })).rejects.toThrow('Invalid parameters')
        })

        it('should drop index and scan/delete keys on deleteAll', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            mockScan.mockResolvedValueOnce(['0', ['doc:test:key1', 'doc:test:key2']])
            await store.delete({ deleteAll: true })
            expect(GlideFt.dropindex).toHaveBeenCalledWith(mockGlideClient, 'test-idx')
            expect(mockScan).toHaveBeenCalled()
            expect(mockDel).toHaveBeenCalledWith(['doc:test:key1', 'doc:test:key2'])
        })

        it('should handle scan returning no keys on deleteAll', async () => {
            mockScan.mockResolvedValueOnce(['0', []])
            await store.delete({ deleteAll: true })
            expect(mockDel).not.toHaveBeenCalled()
        })

        it('should use ClusterScanCursor for GlideClusterClient', async () => {
            const { GlideClusterClient, ClusterScanCursor } = require('@valkey/valkey-glide')
            const mockClusterScan = jest.fn().mockResolvedValue([new ClusterScanCursor(), ['doc:cluster:k1']])
            const mockClusterDel = jest.fn().mockResolvedValue(1)
            const clusterClient = Object.create(GlideClusterClient.prototype)
            clusterClient.scan = mockClusterScan
            clusterClient.del = mockClusterDel
            clusterClient.exec = jest.fn()

            const clusterStore = new ValkeyVectorStore(new FakeEmbeddings(), {
                valkeyClient: clusterClient,
                indexName: 'cluster-idx',
                keyPrefix: 'doc:cluster:'
            })

            await clusterStore.delete({ deleteAll: true })
            expect(mockClusterScan).toHaveBeenCalled()
            expect(mockClusterDel).toHaveBeenCalledWith(['doc:cluster:k1'])
        })
    })

    describe('addVectors', () => {
        it('should throw when no vectors provided', async () => {
            await expect(store.addVectors([], [])).rejects.toThrow('No vectors provided')
        })

        it('should throw when vectors are empty arrays', async () => {
            await expect(store.addVectors([[]], [new Document({ pageContent: '', metadata: {} })])).rejects.toThrow('No vectors provided')
        })

        it('should throw when vectors.length !== documents.length', async () => {
            await expect(
                store.addVectors(
                    [
                        [0.1, 0.2],
                        [0.3, 0.4]
                    ],
                    [new Document({ pageContent: 'only one', metadata: {} })]
                )
            ).rejects.toThrow('Vectors length (2) must match documents length (1)')
        })

        it('should generate UUID-based keys', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockResolvedValue({ numDocs: '0' })

            await store.addVectors([[0.1, 0.2]], [new Document({ pageContent: 'test', metadata: {} })])

            const { Batch } = require('@valkey/valkey-glide')
            const batchInstance = Batch.mock.results[0].value
            const hsetCall = batchInstance.hset.mock.calls[0]
            const key = hsetCall[0]
            expect(key).toMatch(/^doc:test:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
        })

        it('should use user-provided keys when given', async () => {
            await store.addVectors([[0.1, 0.2]], [new Document({ pageContent: 'test', metadata: {} })], {
                keys: ['my-custom-key']
            })

            const { Batch } = require('@valkey/valkey-glide')
            const batchInstance = Batch.mock.results[0].value
            expect(batchInstance.hset).toHaveBeenCalledWith('my-custom-key', expect.any(Object))
        })

        it('should set TTL when configured', async () => {
            const storeWithTTL = new ValkeyVectorStore(new FakeEmbeddings(), {
                valkeyClient: mockGlideClient,
                indexName: 'test-idx',
                keyPrefix: 'doc:test:',
                ttl: 300
            })

            await storeWithTTL.addVectors([[0.1, 0.2]], [new Document({ pageContent: 'test', metadata: {} })])

            const { Batch } = require('@valkey/valkey-glide')
            const batchInstance = Batch.mock.results[0].value
            expect(batchInstance.expire).toHaveBeenCalled()
            expect(batchInstance.expire.mock.calls[0][1]).toBe(300)
        })

        it('should not set TTL when not configured', async () => {
            mockExpire.mockClear()
            await store.addVectors([[0.1, 0.2]], [new Document({ pageContent: 'test', metadata: {} })])

            const { Batch } = require('@valkey/valkey-glide')
            const batchInstance = Batch.mock.results[0].value
            expect(batchInstance.expire).not.toHaveBeenCalled()
        })

        it('should flush batch at exactly batchSize boundary', async () => {
            const docs = Array.from({ length: 3 }, (_, i) => new Document({ pageContent: `doc${i}`, metadata: {} }))
            const vecs = Array.from({ length: 3 }, () => [0.1, 0.2])

            await store.addVectors(vecs, docs, { batchSize: 3 })

            // Exactly one batch exec call (3 docs, batchSize=3)
            expect(mockExec).toHaveBeenCalledTimes(1)
        })

        it('should split into multiple batches when exceeding batchSize', async () => {
            const docs = Array.from({ length: 5 }, (_, i) => new Document({ pageContent: `doc${i}`, metadata: {} }))
            const vecs = Array.from({ length: 5 }, () => [0.1, 0.2])

            await store.addVectors(vecs, docs, { batchSize: 2 })

            // 5 docs with batchSize=2 → 3 exec calls (2, 2, 1)
            expect(mockExec).toHaveBeenCalledTimes(3)
        })

        it('should propagate error when exec fails mid-batch', async () => {
            mockExec.mockRejectedValueOnce(new Error('Connection lost'))

            const docs = Array.from({ length: 3 }, (_, i) => new Document({ pageContent: `doc${i}`, metadata: {} }))
            const vecs = Array.from({ length: 3 }, () => [0.1, 0.2])

            await expect(store.addVectors(vecs, docs, { batchSize: 2 })).rejects.toThrow('Connection lost')
        })
    })

    describe('TTL validation', () => {
        it('should throw for ttl: 0', () => {
            expect(
                () =>
                    new ValkeyVectorStore(new FakeEmbeddings(), {
                        valkeyClient: mockGlideClient,
                        indexName: 'test-idx',
                        ttl: 0
                    })
            ).toThrow('TTL must be a positive integer')
        })

        it('should throw for negative TTL', () => {
            expect(
                () =>
                    new ValkeyVectorStore(new FakeEmbeddings(), {
                        valkeyClient: mockGlideClient,
                        indexName: 'test-idx',
                        ttl: -10
                    })
            ).toThrow('TTL must be a positive integer')
        })

        it('should accept undefined TTL (no expiry)', () => {
            expect(
                () =>
                    new ValkeyVectorStore(new FakeEmbeddings(), {
                        valkeyClient: mockGlideClient,
                        indexName: 'test-idx'
                    })
            ).not.toThrow()
        })
    })

    describe('createIndex', () => {
        it('should not recreate existing index', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockResolvedValue({ numDocs: '0' })
            await store.createIndex(128)
            expect(GlideFt.create).not.toHaveBeenCalled()
        })

        it('should create index with TAG and VECTOR fields', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockRejectedValueOnce(new Error('Unknown index'))
            await store.createIndex(128)
            expect(GlideFt.create).toHaveBeenCalledWith(
                mockGlideClient,
                'test-idx',
                expect.arrayContaining([
                    expect.objectContaining({ type: 'TAG', name: 'metadata_tags' }),
                    expect.objectContaining({ type: 'VECTOR', name: 'content_vector' })
                ]),
                expect.objectContaining({ dataType: 'HASH', prefixes: ['doc:test:'] })
            )
        })

        it('should throw helpful error when TEXT field unsupported', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockRejectedValueOnce(new Error('Unknown index'))
            GlideFt.create.mockRejectedValueOnce(new Error('unknown field type TEXT'))
            await expect(store.createIndex(128)).rejects.toThrow('Valkey Search >= 1.2')
        })

        it('should handle concurrent index creation (already exists)', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockRejectedValueOnce(new Error('Unknown index'))
            GlideFt.create.mockRejectedValueOnce(new Error('Index already exists'))
            // Should not throw — handled gracefully
            await expect(store.createIndex(128)).resolves.toBeUndefined()
        })
    })

    describe('checkIndexExists', () => {
        it('should return true when index exists', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockResolvedValue({ numDocs: '0' })
            expect(await store.checkIndexExists()).toBe(true)
        })

        it('should return false when index does not exist', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockRejectedValueOnce(new Error('Unknown index name'))
            expect(await store.checkIndexExists()).toBe(false)
        })

        it('should throw when search module not loaded', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockRejectedValueOnce(new Error('unknown command FT.INFO'))
            await expect(store.checkIndexExists()).rejects.toThrow('valkey-search module')
        })
    })
})

describe('Valkey_VectorStores node wrapper', () => {
    let node: any

    beforeEach(() => {
        jest.clearAllMocks()
        node = new Valkey_VectorStores()
        mockGetCredentialParam.mockImplementation((key: string) => {
            const map: Record<string, string> = {
                valkeyUrl: 'valkey://localhost:6379',
                valkeyHost: '',
                valkeyPort: '',
                valkeyUser: '',
                valkeyPassword: '',
                valkeyTls: ''
            }
            return map[key] || ''
        })
    })

    describe('init', () => {
        it('should create a vector store and return retriever when output is retriever', async () => {
            const nodeData = {
                credential: 'cred-id',
                inputs: {
                    indexName: 'my-index',
                    embeddings: new FakeEmbeddings(),
                    topK: '5',
                    contentKey: 'content',
                    metadataKey: 'metadata',
                    vectorKey: 'content_vector',
                    valkeyMetadataFilter: undefined
                },
                outputs: { output: 'retriever' }
            }

            const result = await node.init(nodeData, '', {})
            // Retriever has invoke method
            expect(result).toBeDefined()
            expect(result.lc_namespace).toBeDefined()
        })

        it('should return vectorStore when output is vectorStore', async () => {
            const nodeData = {
                credential: 'cred-id',
                inputs: {
                    indexName: 'my-index',
                    embeddings: new FakeEmbeddings(),
                    topK: '3',
                    contentKey: '',
                    metadataKey: '',
                    vectorKey: '',
                    valkeyMetadataFilter: undefined
                },
                outputs: { output: 'vectorStore' }
            }

            const result = await node.init(nodeData, '', {})
            expect(result._vectorstoreType()).toBe('valkey')
            expect((result as any).k).toBe(3)
        })

        it('should parse JSON metadata filter', async () => {
            const nodeData = {
                credential: 'cred-id',
                inputs: {
                    indexName: 'my-index',
                    embeddings: new FakeEmbeddings(),
                    topK: '4',
                    contentKey: '',
                    metadataKey: '',
                    vectorKey: '',
                    valkeyMetadataFilter: '["tag1","tag2"]'
                },
                outputs: { output: 'vectorStore' }
            }

            const result = await node.init(nodeData, '', {})
            expect(result.filter).toEqual(['tag1', 'tag2'])
        })
    })

    describe('vectorStoreMethods.upsert', () => {
        it('should upsert documents and close client', async () => {
            const nodeData = {
                credential: 'cred-id',
                inputs: {
                    indexName: 'upsert-idx',
                    embeddings: new FakeEmbeddings(),
                    replaceIndex: false,
                    contentKey: 'content',
                    metadataKey: 'metadata',
                    vectorKey: 'content_vector',
                    recordManager: undefined,
                    document: [[new Document({ pageContent: 'hello', metadata: {} })]]
                }
            }

            const result = await node.vectorStoreMethods.upsert(nodeData, {})
            expect(result.numAdded).toBe(1)
            expect(mockClose).toHaveBeenCalled()
        })

        it('should drop index when replaceIndex is true', async () => {
            const { GlideFt } = require('@valkey/valkey-glide')
            const nodeData = {
                credential: 'cred-id',
                inputs: {
                    indexName: 'replace-idx',
                    embeddings: new FakeEmbeddings(),
                    replaceIndex: true,
                    contentKey: '',
                    metadataKey: '',
                    vectorKey: '',
                    recordManager: undefined,
                    document: [[new Document({ pageContent: 'test', metadata: {} })]]
                }
            }

            await node.vectorStoreMethods.upsert(nodeData, {})
            expect(GlideFt.dropindex).toHaveBeenCalled()
        })

        it('should filter out documents with empty pageContent', async () => {
            const nodeData = {
                credential: 'cred-id',
                inputs: {
                    indexName: 'filter-idx',
                    embeddings: new FakeEmbeddings(),
                    replaceIndex: false,
                    contentKey: '',
                    metadataKey: '',
                    vectorKey: '',
                    recordManager: undefined,
                    document: [[new Document({ pageContent: 'keep', metadata: {} }), { pageContent: '', metadata: {} }, null]]
                }
            }

            const result = await node.vectorStoreMethods.upsert(nodeData, {})
            expect(result.numAdded).toBe(1)
        })

        it('should close client even on error', async () => {
            mockExec.mockRejectedValueOnce(new Error('exec failed'))
            const { GlideFt } = require('@valkey/valkey-glide')
            GlideFt.info.mockRejectedValueOnce(new Error('Unknown index'))

            const nodeData = {
                credential: 'cred-id',
                inputs: {
                    indexName: 'error-idx',
                    embeddings: new FakeEmbeddings(),
                    replaceIndex: false,
                    contentKey: '',
                    metadataKey: '',
                    vectorKey: '',
                    recordManager: undefined,
                    document: [[new Document({ pageContent: 'test', metadata: {} })]]
                }
            }

            await expect(node.vectorStoreMethods.upsert(nodeData, {})).rejects.toThrow()
            expect(mockClose).toHaveBeenCalled()
        })
    })

    describe('vectorStoreMethods.delete', () => {
        it('should delete by ids and close client', async () => {
            const nodeData = {
                credential: 'cred-id',
                inputs: {
                    indexName: 'del-idx',
                    embeddings: new FakeEmbeddings(),
                    contentKey: '',
                    metadataKey: '',
                    vectorKey: '',
                    recordManager: undefined
                }
            }

            await node.vectorStoreMethods.delete(nodeData, ['key1', 'key2'], {})
            expect(mockDel).toHaveBeenCalled()
            expect(mockClose).toHaveBeenCalled()
        })
    })
})

describe('getConnectionConfig', () => {
    beforeEach(() => {
        jest.clearAllMocks()
    })

    it('should use URL when valkeyUrl is provided', () => {
        mockGetCredentialParam.mockImplementation((key: string) => {
            if (key === 'valkeyUrl') return 'valkey://myhost:6380'
            return ''
        })

        const config = getConnectionConfig({}, { inputs: {} })
        expect(config.host).toBe('myhost')
        expect(config.port).toBe(6380)
    })

    it('should fall back to individual fields when URL is empty', () => {
        mockGetCredentialParam.mockImplementation((key: string) => {
            const map: Record<string, string> = {
                valkeyUrl: '',
                valkeyHost: '10.0.0.1',
                valkeyPort: '6381',
                valkeyUser: 'admin',
                valkeyPassword: 'secret',
                valkeyTls: 'true'
            }
            return map[key] || ''
        })

        const config = getConnectionConfig({}, { inputs: {} })
        expect(config.host).toBe('10.0.0.1')
        expect(config.port).toBe(6381)
        expect(config.username).toBe('admin')
        expect(config.password).toBe('secret')
        expect(config.useTLS).toBe(true)
    })
})
