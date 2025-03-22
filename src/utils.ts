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

export function bufferToTexture(device: GPUDevice, buffer: GPUBuffer): GPUTexture {
  const TEXTURE_WIDTH = Math.min(8192, buffer.size);
  const TEXTURE_HEIGHT = Math.ceil((buffer.size) / TEXTURE_WIDTH);

  const texture = device.createTexture({
    size: {
      width: TEXTURE_WIDTH,
      height: TEXTURE_HEIGHT,
    },
    format: 'r32uint',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  const command = device.createCommandEncoder();
  command.copyBufferToTexture({ buffer }, { texture }, [texture.width, texture.height, texture.depthOrArrayLayers]);
  device.queue.submit([command.finish()]);

  return texture;
}

export const removeValues = (source: string) => source
  .split('\n')
  .filter((line) => !line.toLowerCase().includes('values'))
  .join('\n')
;
