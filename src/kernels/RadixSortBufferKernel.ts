import { AbstractRadixSortKernel, type AbstractRadixSortKernelOptions } from './AbstractRadixSortKernel';
import radixSortSource from '../shaders/RadixSort';
import radixSortSourceLocalShuffle from '../shaders/optimizations/RadixSortLocalShuffle';
import reorderSource from '../shaders/RadixSortReorder';
import { removeValues } from 'src/utils';

export interface BufferKernel {
  keys: GPUBuffer;
  values?: GPUBuffer;
  localShuffle?: boolean;
}

interface RadixSortKernelOptions extends AbstractRadixSortKernelOptions<BufferKernel> {
}

export class RadixSortBufferKernel extends AbstractRadixSortKernel<BufferKernel> {
  public localShuffle: boolean = false;

  constructor(options: RadixSortKernelOptions) {
    super(options);

    this.buffers.keys = options.data.keys;
    if (options.data.values) {
      this.buffers.values = options.data.values;
    }
  }

  get hasValues(): boolean {
    return !!this.data.values;
  }

  protected get blockSumSource(): string {
    const source = this.localShuffle
      ? radixSortSourceLocalShuffle
      : radixSortSource(this.buffers.keys)
    ;

    return this.hasValues ? source : removeValues(source);
  }

  protected get reorderSource(): string {
    return this.hasValues ? reorderSource : removeValues(reorderSource);
  }

  protected createResources(): void {
    // Keys and values double buffering
    this.buffers.tmpKeys = this.device.createBuffer({
      label: 'radix-sort-tmp-keys',
      size: this.count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    if (this.hasValues) {
      this.buffers.tmpValues = this.device.createBuffer({
        label: 'radix-sort-tmp-values',
        size: this.count * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
    }

    // Local Prefix Sum buffer (1 element per item)
    this.buffers.localPrefixSum = this.device.createBuffer({
      label: 'radix-sort-local-prefix-sum',
      size: this.count * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
  }

  protected getPassInData(even: boolean): BufferKernel {
    return {
      keys: even ? this.buffers.keys : this.buffers.tmpKeys,
      values: even ? this.buffers.values : this.buffers.tmpValues,
    };
  }

  protected getPassOutData(even: boolean): BufferKernel {
    return {
      keys: even ? this.buffers.tmpKeys : this.buffers.keys,
      values: even ? this.buffers.tmpValues : this.buffers.values,
    };
  }
}
