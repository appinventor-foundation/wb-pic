export class EnvironmentAdapter {
  async loadImage(imageSource) {
    throw new Error('loadImage must be implemented');
  }

  async createZipArchive(files) {
    throw new Error('createZipArchive must be implemented');
  }

  async saveFile(data, filename) {
    throw new Error('saveFile must be implemented');
  }

  async readZipArchive(data) {
    throw new Error('readZipArchive must be implemented');
  }
}
