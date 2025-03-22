import { RadixSortBufferKernel } from '../src';
import { createBuffers, createTimestampQuery, testRadixSort } from './tests';

const settings = {
  elementCount: 2 ** 20,
  bitCount: 32,
  workgroupSize: 16,
  checkOrder: false,
  localShuffle: false,
  avoidBankConflicts: false,
  sortMode: 'Keys',
  initialSort: 'Random',
  consecutiveSorts: 1,
};

window.onload = async function main() {
  // Create the GPU device
  const adapter = await navigator.gpu?.requestAdapter()
  const device = await adapter?.requestDevice({
    requiredFeatures: ['timestamp-query'],
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: 32 * 32,
    },
  });

  // Check if WebGPU is supported
  if (!device) {
    document.getElementById('info')!.innerHTML = 'WebGPU doesn\'t appear to be supported on this device.';
    document.getElementById('info')!.style.color = '#ff7a48';
    throw new Error('Could not create WebGPU device');
  }

  // Create the Settings panel
  const gui = new GUI(device);

  // Run the tests
  // if (false) {
  //   testRadixSort(device);
  // }

  // Run sort callback
  gui.onClickSort = () => onClickSort(device);
}

/**
 * Run the prefix sum kernel and profile the performance
 */
async function runSort(device: GPUDevice, compareAgainstCpu: boolean = true) {
  const keys = new Uint32Array(settings.elementCount);
  const keysRange = 2 ** settings.bitCount;

  switch (settings.initialSort) {
    case 'Random':
      keys.forEach((_, i) => keys[i] = Math.floor(Math.random() * keysRange));
      break;
    case 'Sorted':
      keys.forEach((_, i) => keys[i] = i);
      break;
  }

  const [keysBuffer] = createBuffers(device, keys);

  let valuesBuffer: GPUBuffer | undefined;

  if (settings.sortMode === 'Keys & Values') {
    const values = new Uint32Array(settings.elementCount).map(() => Math.floor(Math.random() * 1_000_000));
    const buffers = createBuffers(device, values);
    valuesBuffer = buffers[0];
  }

  const kernel = new RadixSortBufferKernel({
    device,
    data: {
      keys: keysBuffer,
      values: valuesBuffer,
    },
    count: settings.elementCount,
    bitCount: settings.bitCount,
    workgroupSize: { x: settings.workgroupSize, y: settings.workgroupSize },
    checkOrder: settings.checkOrder,
    localShuffle: settings.localShuffle,
    avoidBankConflicts: settings.avoidBankConflicts,
  });

  const query = createTimestampQuery(device);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(query.descriptor);
  kernel.dispatch(pass);
  pass.end();

  query.resolve(encoder);
  device.queue.submit([encoder.finish()]);

  const timestamps = await query.getTimestamps();

  const times: {
    cpu: number;
    gpu: number;
  } = {
    cpu: 0,
    gpu: Number(timestamps[1] - timestamps[0]) / 1e6,
  };

  if (compareAgainstCpu) {
    const start = performance.now();
    keys.sort((a, b) => a - b);
    times.cpu = performance.now() - start;
  }

  return times;
}

async function onClickSort(device: GPUDevice) {
  let cpuTime = 0;
  let gpuTime = 0;

  const output = document.getElementById('results')!;
  if (output.children[0].id === 'info') output.innerHTML = '';
  const result = createElm(output, 'div', 'result');

  result.innerHTML += `[${output.children.length}] `;
  result.innerHTML += `Sorting ${makeColor(prettifyNumber(settings.elementCount), '#ff9933')} ${settings.sortMode.toLowerCase()} of ${makeColor(settings.bitCount.toString(), '#ff9933')} bits`;
  result.innerHTML += `<br>Initial sort: ${settings.initialSort}, Workgroup size: ${settings.workgroupSize}x${settings.workgroupSize}`;
  result.innerHTML += `<br>Optimizations: (${settings.checkOrder}, ${settings.localShuffle}, ${settings.avoidBankConflicts})`;

  for (let i = 0; i < settings.consecutiveSorts; i += 1) {
    const times = await runSort(device, i == 0);
    cpuTime += times.cpu;
    gpuTime += times.gpu;
  }

  const gpuAverage = gpuTime / settings.consecutiveSorts;

  result.innerHTML += `<br>> CPU Reference: ${makeColor(cpuTime.toFixed(2) + 'ms', '#abff33')}, `;
  result.innerHTML += `GPU Average (${makeColor(settings.consecutiveSorts.toString(), '#ff9933')} sorts): ${makeColor(gpuAverage.toFixed(2) + 'ms', '#abff33')}, `;
  result.innerHTML += `Speedup: ${makeColor('x' + (cpuTime / gpuAverage).toFixed(2), cpuTime / gpuAverage >= 1 ? '#abff33' : '#ff3333')}<br><br>`;
  result.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

class GUI {
  public dom: HTMLElement;

  public onClickSort: (device: GPUDevice) => void = () => {};

  constructor(device: GPUDevice) {
    this.dom = document.getElementById('gui')!;

    this.createHeader();
    this.createTitle('Radix Sort Kernel');
    this.createSlider(settings, 'elementCount', 'Element Count', settings.elementCount, 1e4, 2 ** 24, 1, { logarithmic: true });
    this.createSlider(settings, 'bitCount', 'Bit Count', settings.bitCount, 4, 32, 4);
    this.createSlider(settings, 'workgroupSize', 'Workgroup Size', settings.workgroupSize, 2, 5, 1, { power_of_two: true });

    this.createTitle('Optimizations');
    this.createCheckbox(settings, 'checkOrder', 'Check If Sorted', settings.checkOrder);
    this.createCheckbox(settings, 'localShuffle', 'Local Shuffle', settings.localShuffle);
    this.createCheckbox(settings, 'avoidBankConflicts', 'Avoid Bank Conflicts', settings.avoidBankConflicts);

    this.createTitle('Testing');
    this.createDropdown(settings, 'initialSort', 'Initial Sort', ['Random', 'Sorted']);
    this.createDropdown(settings, 'sortMode', 'Sort Mode', ['Keys', 'Keys & Values']);
    this.createSlider(settings, 'consecutiveSorts', 'Consecutive Sorts', settings.consecutiveSorts, 1, 20, 1);

    this.createButton('Run Radix Sort', () => this.onClickSort(device));

    this.addHints();
  }

  addHints() {
    const elements = this.dom.querySelectorAll<HTMLElement>('.gui-ctn');
    const hints = [
      '',
      'Number of elements to sort',
      'Number of bits to sort',
      'Workgroup size in x and y dimensions',
      '',
      'Check if the data is sorted after each pass to stop the kernel early',
      'Use local shuffle optimization (does not seem to improve performance)',
      'Avoid bank conflicts in shared memory (does not seem to improve performance)',
      '',
      'Initial order of the elements',
      'Whether to use values in addition to keys',
      'Number of consecutive sorts to run',
      'Run the Radix Sort kernel !',
    ];
    elements.forEach((elm, index) => {
      elm.title = hints[index];
    });
  }

  createTitle(name: string) {
    const ctn = createElm(this.dom, 'div', 'gui-ctn');
    createElm(ctn, 'div', 'gui-title', { innerText: name });
  }

  createHeader() {
    const header = createElm(this.dom, 'div', 'gui-header');
    const icon = createElm(header, 'div', 'gui-icon');
    const link = createElm(header, 'a', 'gui-link', { href: 'https://github.com/kishimisu/WebGPU-Radix-Sort', innerText: 'https://github.com/kishimisu/WebGPU-Radix-Sort' }) as HTMLAnchorElement;
    const githubIcon = `<svg role="img" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><title>GitHub</title><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>`;
    icon.innerHTML = githubIcon;
    link.target = '_blank';
  }

  createSlider(object: Record<string, any>, prop: string, name: string, value = 0, min = 0, max = 1, step = 0.01, { logarithmic = false, power_of_two = false } = {}) {
    const logMin = Math.log(min);
    const logMax = Math.log(max);
    const scale = (logMax - logMin) / (max - min);
    const convert = (x: number) => logarithmic ? Math.round(Math.exp(logMin + scale * (x - min))) : power_of_two ? Math.pow(2, x) : x;
    const convert_back = (x: number) => logarithmic ? (Math.log(x) - logMin) / scale + min : power_of_two ? Math.log2(x) : x;

    const ctn = createElm(this.dom, 'div', 'gui-ctn');
    const label = createElm(ctn, 'label', 'gui-label', { innerText: name });
    const input = createElm(ctn, 'div', 'gui-input');
    const slider = createElm<HTMLInputElement>(input, 'input', 'gui-slider', { type: 'range', min, max, step, value: convert_back(value) });
    const val = createElm(input, 'span', 'gui-value', { innerText: prettifyNumber(value) });

    slider.addEventListener('input', () => {
      let newValue = parseFloat(slider.value);
      if (logarithmic || power_of_two) {
        newValue = convert(parseFloat(slider.value));
      }
      object[prop] = newValue;
      val.innerText = prettifyNumber(newValue);
    })
  }

  createCheckbox(object: Record<string, any>, prop: string, name: string, value: boolean = false) {
    const ctn = createElm(this.dom, 'div', 'gui-ctn');
    const label = createElm(ctn, 'label', 'gui-label', { innerText: name });
    const input = createElm(ctn, 'div', 'gui-input');
    const check = createElm<HTMLInputElement>(input, 'input', 'gui-checkbox', { type: 'checkbox', checked: value });
    const val = createElm(input, 'span', 'gui-value', { innerText: value ? 'true' : 'false' });
    check.addEventListener('change', () => {
      object[prop] = check.checked;
      val.innerText = check.checked ? 'true' : 'false';
    })
  }

  createDropdown(object: Record<string, any>, prop: string, name: string, options: Array<string>) {
    const ctn = createElm(this.dom, 'div', 'gui-ctn');
    const label = createElm(ctn, 'label', 'gui-label', { innerText: name });
    const input = createElm(ctn, 'div', 'gui-input');
    const select = createElm<HTMLSelectElement>(input, 'select', 'gui-select');
    const val = createElm(input, 'span', 'gui-value', { innerText: object[prop] });
    options.forEach((option, index) => {
      const opt = createElm(select, 'option', '', { innerText: option, value: index }) as HTMLOptionElement;
      if (option === object[prop]) opt.selected = true;
    })
    select.addEventListener('change', () => {
      object[prop] = options[parseInt(select.value)];
      val.innerText = options[parseInt(select.value)];
    })
  }

  createButton(name: string, callback: () => void) {
    const ctn = createElm(this.dom, 'div', 'gui-ctn');
    const button = createElm(ctn, 'button', 'gui-button', { innerText: name });
    button.addEventListener('click', callback);
  }
}

function createElm<T = HTMLElement>(parent: HTMLElement, tag: string, className: string, args: Record<string, any> = {}): T {
  const elm = document.createElement(tag);
  elm.className = className;
  Object.keys(args).forEach((key) => {
    // @ts-ignore
    elm[key] = args[key];
  });
  parent.appendChild(elm);
  return elm as T;
}

const makeColor = (val: string, col: string) => `<span style="color:${col}">${val}</span>`;

const prettifyNumber = (x: number) => x.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
