export type WorkgroupSize = {
  x: number;
  y: number;
}

export type DispatchSize = WorkgroupSize;

export function findOptimalDispatchSize(device: GPUDevice, workgroupCount: number) {
  const dispatchSize = { 
    x: workgroupCount, 
    y: 1,
  };

  if (workgroupCount > device.limits.maxComputeWorkgroupsPerDimension) {
    const x = Math.floor(Math.sqrt(workgroupCount));
    const y = Math.ceil(workgroupCount / x);

    dispatchSize.x = x;
    dispatchSize.y = y;
  }

  return dispatchSize;
}

export function createBufferFromData({ device, label, data, usage = 0 }: {
  device: GPUDevice,
  label: string,
  data: number[],
  usage: number,
}) {
  const dispatchSizes = device.createBuffer({
    label,
    usage,
    size: data.length * 4,
    mappedAtCreation: true,
  });

  const dispatchData = new Uint32Array(dispatchSizes.getMappedRange());
  dispatchData.set(data);
  dispatchSizes.unmap();

  return dispatchSizes;
}

export function bufferToTexture(device: GPUDevice, buffer: GPUBuffer, format: GPUTextureFormat): GPUTexture {
  const TEXTURE_WIDTH = Math.min(8192, buffer.size / 4);
  const TEXTURE_HEIGHT = Math.ceil((buffer.size / 4) / TEXTURE_WIDTH);

  const texture = device.createTexture({
    size: {
      width: TEXTURE_WIDTH,
      height: TEXTURE_HEIGHT,
    },
    format,
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  const regex = /^([rgbaRGBA]{1,4})(\d+)(\w+)?$/;
  const matches = format.match(regex)!;
  const components = matches[1].length;
  const command = device.createCommandEncoder();
  command.copyBufferToTexture({
    buffer,
    bytesPerRow: Math.ceil((texture.width * components * 4) / 256) * 256,
  }, { texture }, [texture.width, texture.height, texture.depthOrArrayLayers]);
  device.queue.submit([command.finish()]);

  return texture;
}

export const removeValues = (source: string) => source
  .split('\n')
  .filter((line) => !line.toLowerCase().includes('values'))
  .join('\n')
;
