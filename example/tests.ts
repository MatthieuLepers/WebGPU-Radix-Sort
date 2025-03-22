import { RadixSortKernel, PrefixSumKernel } from '../src';
import type { WorkgroupSize } from '../src/utils';

/** Test the radix sort kernel on GPU for integrity
 * 
 * @param {boolean} keysAndValues - Whether to include a values buffer in the test
 */
export async function testRadixSort(device: GPUDevice, keysAndValues: boolean = false) {
  const {
    maxComputeInvocationsPerWorkgroup,
    maxStorageBufferBindingSize,
    maxBufferSize,
  } = device.limits;

  const maxElements = Math.floor(Math.min(maxBufferSize, maxStorageBufferBindingSize) / 4);
  const workgroupSizes: Array<WorkgroupSize> = [];

  console.log('maxElements:', maxElements);

  const sizes = [2, 4, 8, 16, 32, 64, 128, 256];
  for (let workgroupSizeX of sizes) {
    for (let workgroupSizeY of sizes) {
      if (workgroupSizeX * workgroupSizeY <= maxComputeInvocationsPerWorkgroup) {
        workgroupSizes.push({ x: workgroupSizeX, y: workgroupSizeY });
      }
    }
  }

  for (const workgroupSize of workgroupSizes) {
    for (let exp = 2; exp < 8; exp += 1) {
      const elementCount = Math.floor(Math.min(maxElements, 10 ** exp) * (Math.random() * .1 + .9));
      const subElementCount = Math.floor(elementCount * Math.random() + 1);

      // Create random data
      const bitCount = 32;
      const valueRange = 2 ** bitCount - 1;
      const keys = new Uint32Array(elementCount).map(_ => Math.ceil(Math.random() * valueRange));
      // const keys = new Float32Array(elementCount).map(_ => Math.random() * 10);
      const values = new Uint32Array(elementCount).map((_, i) => i);

      const checkOrder = Math.random() > .5;
      const localShuffle = Math.random() > .5;
      const avoidBankConflicts = Math.random() > .5;

      // Create GPU buffers
      const [keysBuffer, keysBufferMapped] = createBuffers(device, keys);
      const [valuesBuffer, valuesBufferMapped] = createBuffers(device, values);

      // Create kernel
      const kernel = new RadixSortKernel({
        device,
        keys: keysBuffer,
        values: keysAndValues ? valuesBuffer : undefined,
        count: subElementCount,
        bitCount,
        workgroupSize,
        checkOrder,
        localShuffle,
        avoidBankConflicts,
      });

      // Create command buffer and compute pass
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();

      // Run kernel
      kernel.dispatch(pass);
      pass.end();

      // Copy result back to CPU
      encoder.copyBufferToBuffer(kernel.buffers.keys!, 0, keysBufferMapped, 0, elementCount * 4);
      if (keysAndValues) {
        encoder.copyBufferToBuffer(kernel.buffers.values!, 0, valuesBufferMapped, 0, elementCount * 4);
      }

      // Submit command buffer
      device.queue.submit([encoder.finish()]);

      // Read result from GPU
      await keysBufferMapped.mapAsync(GPUMapMode.READ);
      const keysResult = new Uint32Array(keysBufferMapped.getMappedRange().slice());
      keysBufferMapped.unmap();

      // Check result
      const expected = keys.slice(0, subElementCount).sort((a, b) => a - b);
      let isOK = expected.every((v, i) => v === keysResult[i]);

      if (keysAndValues) {
        await valuesBufferMapped.mapAsync(GPUMapMode.READ);
        const valuesResult = new Uint32Array(valuesBufferMapped.getMappedRange().slice());
        valuesBufferMapped.unmap();

        isOK = isOK && valuesResult.every((v, i) => keysResult[i] == keys[v]);
      }

      console.log('Test Radix Sort:', elementCount, subElementCount, workgroupSize, checkOrder, localShuffle, avoidBankConflicts, isOK ? 'OK' : 'ERROR');

      if (!isOK) {
        console.log('keys', keys);
        console.log('keys results', keysResult);
        console.log('keys expected', expected);
        throw new Error('Radix sort error');
      }
    }
  }
}

// Test the prefix sum kernel on GPU
export async function test_prefix_sum(device: GPUDevice) {
  const {
    maxComputeInvocationsPerWorkgroup,
    maxStorageBufferBindingSize,
    maxBufferSize,
  } = device.limits;

  const max_elements = Math.floor(Math.min(maxBufferSize, maxStorageBufferBindingSize) / 4);
  const workgroupSizes: Array<WorkgroupSize> = [];

  const sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256];
  for (let workgroupSizeX of sizes) {
    for (let workgroupSizeY of sizes) {
      if (workgroupSizeX * workgroupSizeY <= maxComputeInvocationsPerWorkgroup) {
        workgroupSizes.push({ x: workgroupSizeX, y: workgroupSizeY });
      }
    }
  }

  for (const workgroupSize of workgroupSizes) {
    for (let exp = 2; exp < 8; exp += 1) {
      const elementCount = Math.floor(Math.min(max_elements, 10 ** exp) * (Math.random() * .1 + .9));
      const subElementCount = Math.floor(elementCount * Math.random() + 1);

      // Create random data
      const data = new Uint32Array(elementCount).map(() => Math.floor(Math.random() * 8));

      // Create GPU buffers
      const [dataBuffer, dataBufferMapped] = createBuffers(device, data);

      // Create kernel
      const prefixSumKernel = new PrefixSumKernel({
        device,
        data: dataBuffer,
        count: subElementCount,
        workgroupSize,
        avoidBankConflicts: false,
      });

      // Create command buffer and compute pass
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();

      // Run kernel
      prefixSumKernel.dispatch(pass);
      pass.end();

      // Copy result back to CPU
      encoder.copyBufferToBuffer(dataBuffer, 0, dataBufferMapped, 0, data.length * 4);

      // Submit command buffer
      device.queue.submit([encoder.finish()]);

      // Read result from GPU
      await dataBufferMapped.mapAsync(GPUMapMode.READ);
      const dataMapped = new Uint32Array(dataBufferMapped.getMappedRange().slice());
      dataBufferMapped.unmap();

      // Check result
      const expected = prefixSumCpu(data.slice(0, subElementCount));
      const isOK = expected.every((v, i) => v === dataMapped[i]);

      console.log('workgroupSize', elementCount, subElementCount, workgroupSize, isOK ? 'OK' : 'ERROR');

      if (!isOK) {
        console.log('input', data);
        console.log('expected', expected);
        console.log('output', dataMapped);
        throw new Error('Prefix sum error');
      }
    }
  }
}

// Create a GPUBuffer with data from an Uint32Array
// Also create a second buffer to read back from GPU
export function createBuffers(device: GPUDevice, data: Uint32Array, usage: number = 0) {
  // Transfer data to GPU
  const dataBuffer = device.createBuffer({
    size: data.length * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | usage,
    mappedAtCreation: true,
  });
  new Uint32Array(dataBuffer.getMappedRange()).set(data);
  dataBuffer.unmap();

  // Create buffer to read back data from CPU
  const dataBufferMapped = device.createBuffer({
    size: data.length * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  return [dataBuffer, dataBufferMapped];
}

// Create a timestamp query object for measuring GPU time
export function createTimestampQuery(device: GPUDevice) {
  const timestampCount = 2
  const querySet = device.createQuerySet({
    type: 'timestamp',
    count: timestampCount,
  });
  const queryBuffer = device.createBuffer({
    size: 8 * timestampCount,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const queryResultBuffer = device.createBuffer({
    size: 8 * timestampCount,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const resolve = (encoder: GPUCommandEncoder) => {
    encoder.resolveQuerySet(querySet, 0, timestampCount, queryBuffer, 0);
    encoder.copyBufferToBuffer(queryBuffer, 0, queryResultBuffer, 0, 8 * timestampCount);
  };

  const getTimestamps = async () => {
    await queryResultBuffer.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(queryResultBuffer.getMappedRange().slice());
    queryResultBuffer.unmap();
    return timestamps;
  }

  return {
    descriptor: {
      timestampWrites: {
        querySet: querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      },
    },
    resolve,
    getTimestamps,
  };
}

// CPU version of the prefix sum algorithm
export function prefixSumCpu(data: Uint32Array) {
  const prefixSum: number[] = [];
  let sum = 0;
  for (let i = 0; i < data.length; i += 1) {
    prefixSum[i] = sum;
    sum += data[i];
  }
  return prefixSum;
}
