/* Filtros IIR por canal para o plot ao vivo (só visualização).
 * Cadeia @125 Hz: notch 60 Hz (rede) + passa-faixa Butterworth 8-30 Hz.
 * Coeficientes gerados por scipy (butter/iirnotch), forma direta II
 * transposta. NÃO toca na janela de inferência (backend tem pipeline próprio).
 */

const BP_SECTIONS = [
  { b: [0.031044877, 0.062089753, 0.031044877], a: [1, -0.444586335, 0.206956889] },
  { b: [1.0, 2.0, 1.0], a: [1, -1.246174093, 0.461009794] },
  { b: [1.0, -2.0, 1.0], a: [1, -0.147584692, 0.60026593] },
  { b: [1.0, -2.0, 1.0], a: [1, -1.66333282, 0.816718384] },
];

const NOTCH_60 = {
  b: [0.94304762, 1.871222816, 0.94304762],
  a: [1, 1.871222816, 0.886095241],
};

function runSection(section, state, x) {
  const { b, a } = section;
  // Transposed Direct Form II: y = b0*x + s0; s0 = b1*x - a1*y + s1; s1 = b2*x - a2*y
  const y = b[0] * x + state[0];
  state[0] = b[1] * x - a[1] * y + state[1];
  state[1] = b[2] * x - a[2] * y;
  return y;
}

class ChannelFilter {
  constructor() {
    this.reset();
  }

  reset() {
    this.notch = [0, 0];
    this.bp = BP_SECTIONS.map(() => [0, 0]);
  }

  process(x) {
    let y = runSection(NOTCH_60, this.notch, x);
    for (let i = 0; i < BP_SECTIONS.length; i += 1) {
      y = runSection(BP_SECTIONS[i], this.bp[i], y);
    }
    return Number.isFinite(y) ? y : 0;
  }
}

export class EegFilterBank {
  constructor(channels = 16) {
    this.channels = channels;
    this.enabled = true;
    this.chain = Array.from({ length: channels }, () => new ChannelFilter());
  }

  setEnabled(on) {
    this.enabled = !!on;
    if (!on) this.reset();
  }

  reset() {
    for (const ch of this.chain) ch.reset();
  }

  // Filtra 1 amostra (array 16ch) e retorna NOVO array; entrada intacta.
  processSample(sample) {
    const out = new Array(this.channels);
    for (let ch = 0; ch < this.channels; ch += 1) {
      const v = Number((sample && sample[ch]) || 0);
      out[ch] = this.enabled ? this.chain[ch].process(v) : v;
    }
    return out;
  }
}
