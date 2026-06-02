/**
 * Integration tests for ValkeyVectorStore.
 *
 * Requirements:
 * - Valkey 9.1+ with valkey-search 1.2+ module loaded
 * - Default: localhost:6379 (override with VALKEY_HOST / VALKEY_PORT)
 *
 * Tests are automatically skipped if Valkey is unavailable.
 */

jest.mock('../../../src/utils', () => ({
    getBaseClasses: jest.fn(() => ['VectorStore']),
    getCredentialData: jest.fn(),
    getCredentialParam: jest.fn()
}))

import { GlideClient } from '@valkey/valkey-glide'
import { Document } from '@langchain/core/documents'

const { ValkeyVectorStore } = require('./Valkey')

const VALKEY_HOST = process.env.VALKEY_HOST || '127.0.0.1'
const VALKEY_PORT = parseInt(process.env.VALKEY_PORT || '6379', 10)

// Deterministic fake embeddings — produces normalized, distinct vectors per text
class FakeEmbeddings {
    async embedDocuments(texts: string[]): Promise<number[][]> {
        return texts.map((t) => this.embed(t))
    }
    async embedQuery(text: string): Promise<number[]> {
        return this.embed(text)
    }
    private embed(text: string): number[] {
        const vec = new Array(128).fill(0)
        for (let i = 0; i < text.length; i++) {
            vec[i % 128] += text.charCodeAt(i) / 1000
        }
        const mag = Math.sqrt(vec.reduce((s: number, v: number) => s + v * v, 0)) || 1
        return vec.map((v: number) => v / mag)
    }
}

async function checkValkeyAvailability(): Promise<string | null> {
    let client: GlideClient | null = null
    try {
        client = await GlideClient.createClient({
            addresses: [{ host: VALKEY_HOST, port: VALKEY_PORT }],
            requestTimeout: 3000
        })
    } catch (err: any) {
        return `Cannot connect to Valkey at ${VALKEY_HOST}:${VALKEY_PORT}: ${err instanceof Error ? err.message : String(err)}`
    }
    try {
        await client.customCommand(['FT._LIST'])
    } catch (err: any) {
        client.close()
        return `valkey-search module not loaded at ${VALKEY_HOST}:${VALKEY_PORT}: ${err instanceof Error ? err.message : String(err)}`
    }
    client.close()
    return null
}

describe('ValkeyVectorStore Integration', () => {
    let client: GlideClient
    let unavailableReason: string | null = null
    const testId = Date.now()
    const indexName = `test-${testId}`

    beforeAll(async () => {
        unavailableReason = await checkValkeyAvailability()
        if (unavailableReason) {
            console.warn(`\n⚠️  Skipping integration tests: ${unavailableReason}\n`)
            return
        }
        client = await GlideClient.createClient({
            addresses: [{ host: VALKEY_HOST, port: VALKEY_PORT }],
            requestTimeout: 5000
        })
    }, 15000)

    afterAll(async () => {
        if (!client) return
        // Clean up all test indices and keys
        try {
            const indices = (await client.customCommand(['FT._LIST'])) as string[]
            for (const idx of indices) {
                if (String(idx).includes(String(testId))) {
                    await client.customCommand(['FT.DROPINDEX', idx]).catch(() => {})
                }
            }
        } catch {
            /* cleanup best-effort */
        }
        try {
            const [, keys] = await client.scan('0', { match: `test:${testId}*`, count: 1000 })
            if (keys.length) await client.del(keys as string[])
        } catch {
            /* cleanup best-effort */
        }
        client.close()
    }, 10000)

    function skip() {
        return !!unavailableReason
    }

    function makeStore(suffix: string, opts: Record<string, any> = {}) {
        return new ValkeyVectorStore(new FakeEmbeddings(), {
            valkeyClient: client,
            indexName: `${indexName}-${suffix}`,
            keyPrefix: `test:${testId}-${suffix}:`,
            ...opts
        })
    }

    describe('index lifecycle', () => {
        it('should return empty results on empty index', async () => {
            if (skip()) return
            const store = makeStore('empty')
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('x'), 5)
            expect(results).toEqual([])
        })

        it('should create index and upsert documents', async () => {
            if (skip()) return
            const store = makeStore('lifecycle')
            await store.addDocuments([
                new Document({ pageContent: 'Valkey is fast', metadata: { source: 'a' } }),
                new Document({ pageContent: 'Vector search is useful', metadata: { source: 'b' } })
            ])
            expect(await store.checkIndexExists()).toBe(true)
        })

        it('should search and return correct documents', async () => {
            if (skip()) return
            const store = makeStore('lifecycle')
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('Valkey is fast'), 2)
            expect(results.length).toBe(2)
            expect(results[0][0].pageContent).toBe('Valkey is fast')
            expect(results[0][0].metadata.source).toBe('a')
        })

        it('should respect top K', async () => {
            if (skip()) return
            const store = makeStore('lifecycle')
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('search'), 1)
            expect(results.length).toBe(1)
        })

        it('should drop and recreate index', async () => {
            if (skip()) return
            const store = makeStore('lifecycle')
            expect(await store.dropIndex()).toBe(true)
            expect(await store.checkIndexExists()).toBe(false)
            await store.addDocuments([new Document({ pageContent: 'recreated', metadata: {} })])
            expect(await store.checkIndexExists()).toBe(true)
        })
    })

    describe('deleteAll cleans up HASH keys', () => {
        it('should remove both index and all HASH keys', async () => {
            if (skip()) return
            const prefix = `test:${testId}-delall:`
            const store = makeStore('delall')
            await store.addDocuments([
                new Document({ pageContent: 'doc1', metadata: {} }),
                new Document({ pageContent: 'doc2', metadata: {} })
            ])

            // Verify keys exist
            const keysBefore: string[] = []
            let cur = '0'
            do {
                const [next, keys] = await client.scan(cur, { match: `${prefix}*`, count: 100 })
                cur = String(next)
                keysBefore.push(...(keys as string[]))
            } while (cur !== '0')
            expect(keysBefore.length).toBe(2)

            await store.delete({ deleteAll: true })

            // Verify index gone
            expect(await store.checkIndexExists()).toBe(false)
            // Verify HASH keys gone
            const keysAfter: string[] = []
            cur = '0'
            do {
                const [next, keys] = await client.scan(cur, { match: `${prefix}*`, count: 100 })
                cur = String(next)
                keysAfter.push(...(keys as string[]))
            } while (cur !== '0')
            expect(keysAfter.length).toBe(0)
        })
    })

    describe('delete by ids', () => {
        it('should delete specific documents by id', async () => {
            if (skip()) return
            const prefix = `test:${testId}-delid:`
            const store = makeStore('delid')
            const keys = [`${prefix}a`, `${prefix}b`, `${prefix}c`]
            await store.addDocuments(
                [
                    new Document({ pageContent: 'doc a', metadata: {} }),
                    new Document({ pageContent: 'doc b', metadata: {} }),
                    new Document({ pageContent: 'doc c', metadata: {} })
                ],
                { keys }
            )

            await store.delete({ ids: [`${prefix}a`, `${prefix}c`] })
            expect(await client.customCommand(['EXISTS', `${prefix}a`])).toBe(0)
            expect(await client.customCommand(['EXISTS', `${prefix}b`])).toBe(1)
            expect(await client.customCommand(['EXISTS', `${prefix}c`])).toBe(0)
        })

        it('should no-op for empty ids array', async () => {
            if (skip()) return
            const store = makeStore('delempty')
            // Should not throw
            await store.delete({ ids: [] })
        })

        it('should not throw for non-existent ids', async () => {
            if (skip()) return
            const store = makeStore('delnoexist')
            await store.delete({ ids: ['nonexistent-key-xyz'] })
        })
    })

    describe('auto-generated UUID keys', () => {
        it('should generate unique keys and not overwrite on concurrent adds', async () => {
            if (skip()) return
            const prefix = `test:${testId}-uuid:`
            const store = makeStore('uuid')

            // Two parallel adds
            await Promise.all([
                store.addDocuments([new Document({ pageContent: 'parallel 1', metadata: {} })]),
                store.addDocuments([new Document({ pageContent: 'parallel 2', metadata: {} })])
            ])

            const [, keys] = await client.scan('0', { match: `${prefix}*`, count: 100 })
            expect(keys.length).toBe(2)
            // Keys should be different
            expect(keys[0]).not.toBe(keys[1])
        })

        it('should produce UUID-format keys', async () => {
            if (skip()) return
            const prefix = `test:${testId}-uuidfmt:`
            const store = makeStore('uuidfmt')
            await store.addDocuments([new Document({ pageContent: 'test', metadata: {} })])

            const [, keys] = await client.scan('0', { match: `${prefix}*`, count: 10 })
            expect(keys.length).toBe(1)
            const key = String(keys[0])
            const uuidPart = key.replace(prefix, '')
            expect(uuidPart).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
        })
    })

    describe('user-provided keys', () => {
        it('should store and retrieve with custom keys', async () => {
            if (skip()) return
            const prefix = `test:${testId}-custom:`
            const store = makeStore('custom')
            const keys = [`${prefix}first`, `${prefix}second`]

            await store.addDocuments(
                [
                    new Document({ pageContent: 'custom key doc 1', metadata: { idx: 0 } }),
                    new Document({ pageContent: 'custom key doc 2', metadata: { idx: 1 } })
                ],
                { keys }
            )

            expect(await client.customCommand(['EXISTS', keys[0]])).toBe(1)
            expect(await client.customCommand(['EXISTS', keys[1]])).toBe(1)
        })
    })

    describe('batch upsert', () => {
        it('should handle more documents than batchSize', async () => {
            if (skip()) return
            const store = makeStore('batch')
            const docs = Array.from({ length: 7 }, (_, i) => new Document({ pageContent: `Batch doc ${i}`, metadata: { i } }))
            await store.addDocuments(docs, { batchSize: 3 })

            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('Batch doc'), 7)
            expect(results.length).toBe(7)
        })
    })

    describe('TTL support', () => {
        it('should set TTL on all documents', async () => {
            if (skip()) return
            const prefix = `test:${testId}-ttl:`
            const store = makeStore('ttl', { ttl: 120 })
            await store.addDocuments([
                new Document({ pageContent: 'ttl doc 1', metadata: {} }),
                new Document({ pageContent: 'ttl doc 2', metadata: {} })
            ])

            // Collect all keys via scan loop
            const allKeys: string[] = []
            let cursor = '0'
            do {
                const [next, keys] = await client.scan(cursor, { match: `${prefix}*`, count: 100 })
                cursor = String(next)
                allKeys.push(...(keys as string[]))
            } while (cursor !== '0')

            expect(allKeys.length).toBe(2)
            for (const key of allKeys) {
                const ttl = await client.ttl(key)
                expect(ttl).toBeGreaterThan(0)
                expect(ttl).toBeLessThanOrEqual(120)
            }
        })

        it('should not set TTL when not configured', async () => {
            if (skip()) return
            const prefix = `test:${testId}-nottl:`
            const store = makeStore('nottl')
            await store.addDocuments([new Document({ pageContent: 'no ttl doc', metadata: {} })])

            const [, keys] = await client.scan('0', { match: `${prefix}*`, count: 10 })
            expect(keys.length).toBe(1)
            const ttl = await client.ttl(keys[0] as string)
            expect(ttl).toBe(-1) // -1 means no expiry
        })
    })

    describe('metadata filtering', () => {
        it('should filter results by metadata content', async () => {
            if (skip()) return
            const store = makeStore('filter')
            await store.addDocuments([
                new Document({ pageContent: 'apple fruit', metadata: { category: 'fruit' } }),
                new Document({ pageContent: 'banana fruit', metadata: { category: 'fruit' } }),
                new Document({ pageContent: 'carrot vegetable', metadata: { category: 'vegetable' } })
            ])

            const queryVec = await new FakeEmbeddings().embedQuery('food')
            // Filter for 'fruit' — TEXT search within metadata JSON
            const results = await store.similaritySearchVectorWithScore(queryVec, 3, ['fruit'])
            expect(results.length).toBe(2)
            for (const [doc] of results) {
                expect(doc.metadata.category).toBe('fruit')
            }
        })

        it('should support OR filtering with multiple terms', async () => {
            if (skip()) return
            const store = makeStore('filter') // reuse index from above
            const queryVec = await new FakeEmbeddings().embedQuery('food')
            const results = await store.similaritySearchVectorWithScore(queryVec, 3, ['fruit', 'vegetable'])
            expect(results.length).toBe(3)
        })

        it('should return no results when filter matches nothing', async () => {
            if (skip()) return
            const store = makeStore('filter')
            const queryVec = await new FakeEmbeddings().embedQuery('food')
            const results = await store.similaritySearchVectorWithScore(queryVec, 3, ['nonexistent'])
            expect(results.length).toBe(0)
        })

        it('should throw when both instance and method filter provided', async () => {
            if (skip()) return
            const store = makeStore('filterboth', { filter: ['tag1'] })
            const queryVec = await new FakeEmbeddings().embedQuery('x')
            await expect(store.similaritySearchVectorWithScore(queryVec, 1, ['tag2'])).rejects.toThrow('cannot provide both')
        })

        it('should apply constructor filter on search', async () => {
            if (skip()) return
            const store = new ValkeyVectorStore(new FakeEmbeddings(), {
                valkeyClient: client,
                indexName: `${indexName}-filter`, // reuse existing index with data
                keyPrefix: `test:${testId}-filter:`,
                filter: ['vegetable']
            })
            const queryVec = await new FakeEmbeddings().embedQuery('food')
            const results = await store.similaritySearchVectorWithScore(queryVec, 3)
            expect(results.length).toBe(1)
            expect(results[0][0].metadata.category).toBe('vegetable')
        })
    })

    describe('search result ordering', () => {
        it('should return most similar document first with ascending scores', async () => {
            if (skip()) return
            const store = makeStore('order')
            await store.addDocuments([
                new Document({ pageContent: 'completely unrelated cooking recipe for pasta', metadata: {} }),
                new Document({ pageContent: 'Valkey vector similarity search', metadata: {} }),
                new Document({ pageContent: 'random text about weather forecast', metadata: {} })
            ])

            const queryVec = await new FakeEmbeddings().embedQuery('Valkey vector similarity search')
            const results = await store.similaritySearchVectorWithScore(queryVec, 3)
            expect(results.length).toBe(3)
            expect(results[0][0].pageContent).toContain('Valkey vector')
            // Scores ascending (lower = more similar for cosine distance)
            for (let i = 0; i < results.length - 1; i++) {
                expect(results[i][1]).toBeLessThanOrEqual(results[i + 1][1])
            }
        })
    })

    describe('metadata preservation', () => {
        it('should preserve complex metadata through upsert and search', async () => {
            if (skip()) return
            const store = makeStore('meta')
            const meta = { source: 'test', page: 42, tags: ['a', 'b'], nested: { key: 'value' } }
            await store.addDocuments([new Document({ pageContent: 'metadata test', metadata: meta })])

            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('metadata test'), 1)
            expect(results.length).toBe(1)
            expect(results[0][0].metadata).toEqual(meta)
        })
    })

    describe('static factory methods', () => {
        it('fromDocuments should create and populate store', async () => {
            if (skip()) return
            const store = await ValkeyVectorStore.fromDocuments(
                [new Document({ pageContent: 'factory doc', metadata: { via: 'factory' } })],
                new FakeEmbeddings(),
                { valkeyClient: client, indexName: `${indexName}-factory`, keyPrefix: `test:${testId}-factory:` }
            )
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('factory doc'), 1)
            expect(results.length).toBe(1)
            expect(results[0][0].metadata.via).toBe('factory')
        })

        it('fromTexts should create store with array metadata', async () => {
            if (skip()) return
            const store = await ValkeyVectorStore.fromTexts(['text one', 'text two'], [{ idx: 0 }, { idx: 1 }], new FakeEmbeddings(), {
                valkeyClient: client,
                indexName: `${indexName}-fromtxt`,
                keyPrefix: `test:${testId}-fromtxt:`
            })
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('text one'), 2)
            expect(results.length).toBe(2)
        })

        it('fromTexts should apply single metadata object to all docs', async () => {
            if (skip()) return
            const store = await ValkeyVectorStore.fromTexts(['a', 'b'], { shared: true }, new FakeEmbeddings(), {
                valkeyClient: client,
                indexName: `${indexName}-fromtxt2`,
                keyPrefix: `test:${testId}-fromtxt2:`
            })
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('a'), 2)
            for (const [doc] of results) {
                expect(doc.metadata.shared).toBe(true)
            }
        })
    })

    describe('edge cases', () => {
        it('should handle documents with empty pageContent', async () => {
            if (skip()) return
            const store = makeStore('edgeempty')
            await store.addDocuments([new Document({ pageContent: '', metadata: { empty: true } })])
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery(''), 1)
            expect(results.length).toBe(1)
            expect(results[0][0].pageContent).toBe('')
            expect(results[0][0].metadata.empty).toBe(true)
        })

        it('should handle documents with very long content', async () => {
            if (skip()) return
            const store = makeStore('edgelong')
            const longContent = 'x'.repeat(10000)
            await store.addDocuments([new Document({ pageContent: longContent, metadata: {} })])
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery(longContent), 1)
            expect(results.length).toBe(1)
            expect(results[0][0].pageContent.length).toBe(10000)
        })

        it('should handle metadata with special characters', async () => {
            if (skip()) return
            const store = makeStore('edgespecial')
            const meta = { key: 'value with "quotes" and {braces}', unicode: '日本語' }
            await store.addDocuments([new Document({ pageContent: 'special chars', metadata: meta })])
            const results = await store.similaritySearchVectorWithScore(await new FakeEmbeddings().embedQuery('special chars'), 1)
            expect(results[0][0].metadata).toEqual(meta)
        })
    })
})
