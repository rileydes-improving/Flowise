import { INodeParams, INodeCredential } from '../src/Interface'

class ValkeyUrlApi implements INodeCredential {
    label: string
    name: string
    version: number
    description: string
    inputs: INodeParams[]

    constructor() {
        this.label = 'Valkey URL'
        this.name = 'valkeyUrlApi'
        this.version = 1.0
        this.description = 'Connect to Valkey using a connection URL'
        this.inputs = [
            {
                label: 'Valkey URL',
                name: 'valkeyUrl',
                type: 'password',
                placeholder: 'valkey://user:password@localhost:6379'
            }
        ]
    }
}

module.exports = { credClass: ValkeyUrlApi }
