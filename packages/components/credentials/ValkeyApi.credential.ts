import { INodeParams, INodeCredential } from '../src/Interface'

class ValkeyApi implements INodeCredential {
    label: string
    name: string
    version: number
    description: string
    inputs: INodeParams[]

    constructor() {
        this.label = 'Valkey API'
        this.name = 'valkeyApi'
        this.version = 1.0
        this.description = 'Connect to Valkey using host, port, and optional authentication'
        this.inputs = [
            {
                label: 'Valkey Host',
                name: 'valkeyHost',
                type: 'string',
                default: '127.0.0.1'
            },
            {
                label: 'Port',
                name: 'valkeyPort',
                type: 'number',
                default: '6379'
            },
            {
                label: 'User',
                name: 'valkeyUser',
                type: 'string',
                placeholder: '<VALKEY_USERNAME>',
                optional: true
            },
            {
                label: 'Password',
                name: 'valkeyPassword',
                type: 'password',
                placeholder: '<VALKEY_PASSWORD>',
                optional: true
            },
            {
                label: 'Use TLS',
                name: 'valkeyTls',
                type: 'boolean',
                optional: true
            }
        ]
    }
}

module.exports = { credClass: ValkeyApi }
