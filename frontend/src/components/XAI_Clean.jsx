import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';

// Clean, minimal XAI components - no fancy animations, just entry fade-in

export const IntegratedGradientsCard = ({ data, accentColor, delay, audioDuration, isFake }) => {
  // Backend returns 'feature_attribution', not 'temporal_attribution'
  if (!data?.feature_attribution) {
    return null;
  }
  const scores = data.feature_attribution;
  if (!Array.isArray(scores) || scores.length === 0) {
    return null;
  }
  const maxAttr = Math.max(...scores.map(Math.abs), 0.001);
  
  // Use actual audio duration if provided, otherwise estimate
  const actualDuration = audioDuration || (scores.length * 0.05);
  const maxTime = actualDuration;

  // Normalize scores
  const normalized = scores.map(s => Math.abs(s) / maxAttr);

  // Count HIGH ATTRIBUTION REGIONS (continuous segments above threshold)
  // Higher threshold = only the MOST influential segments
  let peakRegions = 0;
  let inPeakRegion = false;
  const threshold = 0.75; // Increased from 0.6 to be more selective
  
  normalized.forEach((norm, i) => {
    if (norm > threshold) {
      if (!inPeakRegion) {
        peakRegions++;
        inPeakRegion = true;
      }
    } else {
      inPeakRegion = false;
    }
  });

  // Determine what high attribution means based on prediction
  const highAttrMeaning = isFake 
    ? "contributed most to FAKE detection" 
    : "contributed most to REAL classification";
  const highAttrColor = accentColor; // Use prediction color (red for fake, green for real)

  const nBins = 50;
  const binSize = Math.max(1, Math.floor(normalized.length / nBins));
  const binned = Array.from({ length: nBins }, (_, i) => {
    const start = i * binSize;
    const end = Math.min(normalized.length, start + binSize);
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j++) {
      const v = normalized[j];
      if (typeof v !== 'number' || !Number.isFinite(v)) continue;
      sum += v;
      count++;
    }
    const mean = count > 0 ? sum / count : 0;
    const x = i;
    return {
      bin: x,
      Attribution: mean
    };
  });

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '40px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>Integrated Gradients - Feature Attribution</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '800px' }}>
            Shows <span style={{ color: accentColor, fontWeight: '700' }}>which feature bins</span> {highAttrMeaning}.
            Each bin aggregates a slice of the embedding attribution vector.
            <br /><span style={{ fontSize: '13px', color: '#666', fontStyle: 'italic', marginTop: '8px', display: 'block' }}>
              Bins 0-24 summarize <strong>mean</strong>-derived features; bins 25-49 summarize <strong>std</strong>-derived features.
            </span>
          </p>
        </div>

        <div style={{ background: 'rgba(0,0,0,0.4)', borderRadius: '16px', padding: '28px 24px 20px', border: '1px solid rgba(255,255,255,0.08)', position: 'relative', overflow: 'hidden', boxShadow: 'inset 0 2px 12px rgba(0,0,0,0.3)' }}>
          <ResponsiveContainer width="100%" height={360}>
            <AreaChart data={binned} margin={{ top: 18, right: 26, left: 0, bottom: 10 }}>
              <defs>
                <linearGradient id="igArea" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={accentColor} stopOpacity={0.55} />
                  <stop offset="55%" stopColor={accentColor} stopOpacity={0.18} />
                  <stop offset="100%" stopColor={accentColor} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="1 3" stroke="rgba(255,255,255,0.03)" vertical={false} />
              <ReferenceLine x={24.5} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 4" />
              <XAxis
                dataKey="bin"
                stroke="#444"
                tick={{ fill: '#999' }}
                axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                tickLine={{ stroke: '#444' }}
                ticks={[0, 24, 25, 49]}
                tickFormatter={(v) => {
                  if (v === 0) return 'μ0';
                  if (v === 24) return 'μ24';
                  if (v === 25) return 'σ0';
                  if (v === 49) return 'σ24';
                  return String(v);
                }}
              />
              <YAxis
                stroke="#444"
                tick={{ fill: '#999' }}
                axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                tickLine={{ stroke: '#444' }}
                domain={[0, 1]}
                tickFormatter={(v) => `${Math.round(v * 100)}%`}
              />
              <Tooltip
                contentStyle={{
                  background: 'rgba(5, 5, 5, 0.97)',
                  backdropFilter: 'blur(12px)',
                  border: '1px solid rgba(167, 139, 250, 0.2)',
                  borderRadius: '12px',
                  color: '#fff',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)',
                  fontSize: '13px',
                  padding: '12px 16px'
                }}
                labelStyle={{ color: '#a78bfa', fontWeight: '700', marginBottom: '8px', fontSize: '14px' }}
                labelFormatter={(v) => {
                  const bin = typeof v === 'number' ? v : Number(v);
                  if (!Number.isFinite(bin)) return 'Bin';
                  const kind = bin < 25 ? 'Mean (μ)' : 'Std (σ)';
                  const localIdx = bin < 25 ? bin : (bin - 25);
                  return `${kind} bin ${localIdx}`;
                }}
                formatter={(value) => [
                  typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : value,
                  'Attribution'
                ]}
              />
              <Area type="monotone" dataKey="Attribution" stroke={accentColor} strokeWidth={3} fill="url(#igArea)" dot={false} activeDot={{ r: 5, fill: accentColor, stroke: '#fff', strokeWidth: 2 }} />
            </AreaChart>
          </ResponsiveContainer>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.06)', color: '#666', fontSize: '13px', fontWeight: '700' }}>
            <span>μ bins (0-24)</span>
            <span style={{ color: '#888' }}>FEATURE BINS →</span>
            <span>σ bins (25-49)</span>
          </div>
        </div>

        {/* Stats */}
        <div style={{ marginTop: '32px', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
          <div style={{ padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
            <div style={{ color: '#888', fontSize: '11px', marginBottom: '8px', fontWeight: '700' }}>MAX ATTRIBUTION</div>
            <div style={{ color: '#fff', fontSize: '32px', fontWeight: '900' }}>{(Math.max(...normalized) * 100).toFixed(0)}%</div>
          </div>
          <div style={{ padding: '20px', background: `${accentColor}10`, borderRadius: '12px', border: `1px solid ${accentColor}30`, textAlign: 'center' }}>
            <div style={{ color: '#888', fontSize: '11px', marginBottom: '8px', fontWeight: '700' }}>HIGH-ATTRIBUTION REGIONS</div>
            <div style={{ color: accentColor, fontSize: '32px', fontWeight: '900' }}>{peakRegions}</div>
            <div style={{ color: '#666', fontSize: '10px', marginTop: '4px' }}>Above-threshold segments</div>
          </div>
          <div style={{ padding: '20px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)', textAlign: 'center' }}>
            <div style={{ color: '#888', fontSize: '11px', marginBottom: '8px', fontWeight: '700' }}>AVG ATTRIBUTION</div>
            <div style={{ color: '#fff', fontSize: '32px', fontWeight: '900' }}>{(normalized.reduce((a, b) => a + b, 0) / normalized.length * 100).toFixed(0)}%</div>
          </div>
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};


export const MelSpectrogramCard = ({ data, accentColor, delay, isFake }) => {
  if (!data?.mel_spectrogram || !Array.isArray(data.mel_spectrogram) || data.mel_spectrogram.length === 0) {
    return null;
  }

  const mel = data.mel_spectrogram;
  const nFreq = mel.length;
  const nTime = Array.isArray(mel[0]) ? mel[0].length : 0;
  if (nFreq === 0 || nTime === 0) return null;

  const suspicious = Array.isArray(data.suspicious_bands) ? data.suspicious_bands : [];

  // Robust percentile range for contrast
  const values = [];
  for (let i = 0; i < nFreq; i++) {
    const row = mel[i];
    if (!Array.isArray(row)) continue;
    for (let j = 0; j < row.length; j++) {
      const v = row[j];
      if (typeof v !== 'number' || !Number.isFinite(v)) continue;
      values.push(v);
    }
  }
  values.sort((a, b) => a - b);
  const q = (p) => {
    if (values.length === 0) return 0;
    const idx = Math.min(values.length - 1, Math.max(0, Math.floor(p * (values.length - 1))));
    return values[idx];
  };
  let mn = q(0.02);
  let mx = q(0.98);
  if (!Number.isFinite(mn) || !Number.isFinite(mx) || mx <= mn) {
    mn = -80;
    mx = 0;
  }

  // Render with native resolution and smooth scaling
  const canvasRef = React.useRef(null);
  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = 980;
    const h = 360;
    canvas.width = w;
    canvas.height = h;

    const srcW = nTime;
    const srcH = nFreq;
    const off = document.createElement('canvas');
    off.width = srcW;
    off.height = srcH;
    const offCtx = off.getContext('2d');
    if (!offCtx) return;
    const img = offCtx.createImageData(srcW, srcH);
    const dataArr = img.data;

    const clamp01 = (x) => Math.max(0, Math.min(1, x));
    const cmap = (t) => {
      // viridis-ish
      const stops = [
        [0.0, [8, 5, 16]],
        [0.25, [86, 20, 148]],
        [0.5, [190, 58, 168]],
        [0.75, [238, 120, 208]],
        [1.0, [252, 238, 255]]
      ];
      const tt = clamp01(t);
      let a = stops[0];
      let b = stops[stops.length - 1];
      for (let i = 0; i < stops.length - 1; i++) {
        if (tt >= stops[i][0] && tt <= stops[i + 1][0]) {
          a = stops[i];
          b = stops[i + 1];
          break;
        }
      }
      const span = Math.max(1e-9, b[0] - a[0]);
      const u = (tt - a[0]) / span;
      const r = Math.round(a[1][0] + (b[1][0] - a[1][0]) * u);
      const g = Math.round(a[1][1] + (b[1][1] - a[1][1]) * u);
      const bb = Math.round(a[1][2] + (b[1][2] - a[1][2]) * u);
      return [r, g, bb];
    };

    for (let f = 0; f < srcH; f++) {
      const row = mel[f];
      for (let t = 0; t < srcW; t++) {
        const v = Array.isArray(row) ? row[t] : mn;
        const norm = clamp01((v - mn) / (mx - mn));
        const gamma = Math.pow(norm, 0.75);
        const [r, g, b] = cmap(gamma);

        // Flip vertical for display (low freq at bottom)
        const y = (srcH - 1 - f);
        const idx = (y * srcW + t) * 4;
        dataArr[idx + 0] = r;
        dataArr[idx + 1] = g;
        dataArr[idx + 2] = b;
        dataArr[idx + 3] = 255;
      }
    }
    offCtx.putImageData(img, 0, 0);

    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, w, h);

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
      const yy = (h * i) / 5;
      ctx.beginPath();
      ctx.moveTo(0, yy);
      ctx.lineTo(w, yy);
      ctx.stroke();
    }
  }, [accentColor, mn, mx, nFreq, nTime]);

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '28px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>Mel Spectrogram</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '850px' }}>
            Acoustic fingerprint of the audio. Synthetic speech often shows <span style={{ color: accentColor, fontWeight: '700' }}>unnaturally smooth</span> energy distribution and low-variance bands.
          </p>
        </div>

        <div style={{
          background: 'rgba(0,0,0,0.55)',
          borderRadius: '16px',
          padding: '18px',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow: 'inset 0 2px 12px rgba(0,0,0,0.35)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <div style={{ color: '#aaa', fontSize: '12px', fontWeight: '700', letterSpacing: '0.5px' }}>TIME →</div>
            <div style={{ color: '#666', fontSize: '12px', fontWeight: '700', fontFamily: 'monospace' }}>
              dB range: {mn.toFixed(0)} to {mx.toFixed(0)}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '12px', alignItems: 'stretch' }}>
            <div style={{
              writingMode: 'vertical-rl',
              transform: 'rotate(180deg)',
              color: '#aaa',
              fontSize: '12px',
              fontWeight: '800',
              letterSpacing: '0.5px',
              padding: '6px 0'
            }}>
              FREQUENCY ↑
            </div>

            <div style={{ position: 'relative' }}>
              <canvas
                ref={canvasRef}
                style={{
                  width: '100%',
                  maxHeight: '360px',
                  borderRadius: '12px',
                  display: 'block'
                }}
              />
              <div style={{
                position: 'absolute',
                inset: 0,
                borderRadius: '12px',
                pointerEvents: 'none',
                boxShadow: `0 0 0 1px rgba(255,255,255,0.06), 0 0 0 2px ${accentColor}10`
              }} />
            </div>
          </div>

          <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
            <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px' }}>LOW</div>
            <div style={{
              flex: 1,
              height: '10px',
              borderRadius: '999px',
              background: 'linear-gradient(90deg, rgb(8, 5, 16), rgb(86, 20, 148), rgb(190, 58, 168), rgb(238, 120, 208), rgb(252, 238, 255))',
              border: '1px solid rgba(255,255,255,0.08)'
            }} />
            <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px' }}>HIGH</div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', color: '#666', fontSize: '12px', fontWeight: '700' }}>
            <span>Start</span>
            <span>End</span>
          </div>
        </div>

        <div style={{ marginTop: '24px' }}>
          <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px', marginBottom: '10px' }}>
            SUSPICIOUS BANDS (LOW VARIANCE)
          </div>
          {suspicious.length > 0 ? (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px' }}>
              {suspicious.slice(0, 6).map((b, i) => (
                <div key={i} style={{
                  padding: '14px 16px',
                  borderRadius: '12px',
                  background: `${accentColor}10`,
                  border: `1px solid ${accentColor}25`,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div>
                    <div style={{ color: '#fff', fontSize: '13px', fontWeight: '800' }}>{b.freq_range || 'N/A'}</div>
                    <div style={{ color: '#888', fontSize: '12px' }}>uniformity: {typeof b.uniformity === 'number' ? b.uniformity.toFixed(3) : 'N/A'}</div>
                  </div>
                  <div style={{
                    padding: '6px 10px',
                    borderRadius: '999px',
                    background: 'rgba(0,0,0,0.35)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    color: accentColor,
                    fontSize: '11px',
                    fontWeight: '900',
                    letterSpacing: '0.5px'
                  }}>
                    {isFake ? 'FAKE' : 'REAL'}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div style={{
              padding: '16px',
              borderRadius: '12px',
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.06)',
              color: '#888',
              fontSize: '13px'
            }}>
              No suspicious frequency bands detected.
            </div>
          )}
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};

export const SHAPCard = ({ data, accentColor, delay, isFake }) => {
  if (!data) return null;
  const expertContrib = data?.expert_contributions && typeof data.expert_contributions === 'object'
    ? data.expert_contributions
    : {};
  const expertEntries = Object.entries(expertContrib);
  const maxAbs = Math.max(
    1e-9,
    ...expertEntries.map(([, v]) => (typeof v === 'number' && Number.isFinite(v) ? Math.abs(v) : 0))
  );
  const baselinePred = typeof data?.baseline_pred === 'number' ? data.baseline_pred : null;
  const actualPred = typeof data?.actual_pred === 'number' ? data.actual_pred : null;

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '40px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>SHAP Expert Contributions</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '800px' }}>
            Shows <span style={{ color: '#3b82f6', fontWeight: '700' }}>how much each expert (HuBERT / Wav2Vec2)</span> contributed to the final <span style={{ fontWeight: '700' }}>{isFake ? 'FAKE' : 'REAL'}</span> probability.
            Positive values push <span style={{ fontWeight: '700' }}>toward FAKE</span>; negative values push <span style={{ fontWeight: '700' }}>toward REAL</span>.
          </p>
          {(baselinePred !== null && actualPred !== null) && (
            <div style={{
              marginTop: '16px',
              padding: '14px 16px',
              background: 'rgba(255,255,255,0.02)',
              borderRadius: '12px',
              border: '1px solid rgba(255,255,255,0.06)',
              color: '#aaa',
              fontSize: '13px',
              fontFamily: 'monospace',
              display: 'flex',
              justifyContent: 'space-between',
              gap: '12px'
            }}>
              <span>baseline P(fake)={baselinePred.toFixed(3)}</span>
              <span>actual P(fake)={actualPred.toFixed(3)}</span>
            </div>
          )}
        </div>

        {/* Expert Contributions */}
        <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
          {expertEntries.length > 0 ? expertEntries.map(([name, val], i) => {
            const value = typeof val === 'number' ? val : 0;
            const isPos = value > 0;
            const posColor = '#ef4444';
            const negColor = '#22c55e';
            const magnitude = Math.min(1, Math.abs(value) / maxAbs);
            const halfW = 150;
            const barW = Math.max(2, Math.round(halfW * magnitude));
            return (
              <div key={i} style={{
                background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)',
                borderRadius: '12px', padding: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center'
              }}>
                <div>
                  <div style={{ color: '#fff', fontSize: '15px', fontWeight: '700', marginBottom: '4px' }}>
                    {name.includes('hubert') ? 'HuBERT' : 'Wav2Vec2'}
                  </div>
                  <div style={{ color: '#666', fontSize: '12px' }}>
                    {name.includes('hubert') ? 'Acoustic Expert' : 'Linguistic Expert'}
                  </div>
                  <div style={{ marginTop: '12px' }}>
                    <div style={{
                      width: `${halfW * 2}px`,
                      height: '10px',
                      borderRadius: '999px',
                      background: 'rgba(255,255,255,0.04)',
                      border: '1px solid rgba(255,255,255,0.06)',
                      position: 'relative',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        position: 'absolute',
                        left: '50%',
                        top: 0,
                        bottom: 0,
                        width: '1px',
                        background: 'rgba(255,255,255,0.14)'
                      }} />
                      <div style={{
                        position: 'absolute',
                        top: '1px',
                        height: '8px',
                        borderRadius: '999px',
                        background: isPos ? posColor : negColor,
                        opacity: 0.9,
                        left: isPos ? `calc(50% + 1px)` : `calc(50% - ${barW}px)`,
                        width: `${barW}px`
                      }} />
                    </div>
                    <div style={{ marginTop: '6px', color: '#666', fontSize: '11px', fontWeight: '700' }}>
                      {isPos ? 'pushes toward FAKE' : 'pushes toward REAL'}
                    </div>
                  </div>
                </div>
                <div style={{ color: isPos ? posColor : negColor, fontSize: '24px', fontWeight: '900' }}>
                  {isPos ? '+' : ''}{value.toFixed(3)}
                </div>
              </div>
            );
          }) : (
            <div style={{
              gridColumn: '1 / -1',
              padding: '18px',
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.05)',
              borderRadius: '12px',
              color: '#888',
              fontSize: '13px'
            }}>
              No expert SHAP contributions available.
            </div>
          )}
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};

// LRP REMOVED PER USER REQUEST

export const LRPCard = ({ data, accentColor, delay }) => {
  return null; // Component disabled
};

// BACKUP (not rendered):
const LRPCardBackup = ({ data, accentColor, delay }) => {
  if (!data?.temporal_relevance) return null;
  const scores = data.temporal_relevance;
  const maxRel = Math.max(...scores.map(Math.abs), 0.001);

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: '#0f0f0f', border: '1px solid #2a2a2a', borderRadius: '20px', padding: '40px', marginBottom: '24px' }}>
        <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '800', marginBottom: '8px' }}>Layer-wise Relevance</h3>
        <p style={{ color: '#888', fontSize: '14px', marginBottom: '32px' }}>
          Feature relevance heatmap. Brighter = more relevant to prediction.
        </p>
        <div style={{ background: '#000', borderRadius: '12px', padding: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1px', height: '220px' }}>
            {scores.map((score, i) => {
              const norm = Math.abs(score) / maxRel;
              const color = norm > 0.8 ? '#ec4899' : norm > 0.6 ? '#a855f7' : norm > 0.4 ? '#8b5cf6' : '#3b82f6';
              return (
                <div key={i} style={{
                  flex: 1, height: `${Math.max(norm * 100, 2)}%`, minWidth: '2px', maxWidth: '4px',
                  background: color, borderRadius: '1px'
                }} />
              );
            })}
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', color: '#666', fontSize: '12px' }}>
            <span>Start</span><span>End</span>
          </div>
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};

export const ImprovedWaveformCard = ({ data, accentColor, delay, isFake }) => {
  if (!data?.timestamps) return null;
  const { timestamps, scores } = data;
  const maxTime = timestamps[timestamps.length - 1] || 1;
  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  
  // Generate data-driven waveform from temporal scores (no random/synthetic-only shape)
  const samples = 250;
  const deltas = scores.map(s => s - avgScore);
  const maxDelta = Math.max(...deltas.map(v => Math.abs(v)), 0.0001);
  const wave = Array.from({ length: samples }, (_, i) => {
    const pos = (i / (samples - 1)) * (scores.length - 1);
    const idx0 = Math.floor(pos);
    const idx1 = Math.min(idx0 + 1, scores.length - 1);
    const frac = pos - idx0;
    const s = scores[idx0] * (1 - frac) + scores[idx1] * frac;
    const delta = s - avgScore;
    // Center around 0 and normalize to [-1, 1]
    return Math.max(-1, Math.min(1, delta / maxDelta));
  });

  // Count high-score regions (above average)
  const highScoreRegions = scores.filter(s => s > avgScore).length;
  const highScorePercent = (highScoreRegions / scores.length * 100).toFixed(0);
  
  // Context-aware labels
  const highlightMeaning = isFake 
    ? "segments where FAKE score was highest (model was most confident it's fake here)"
    : "segments where REAL confidence was highest (model was most confident it's real here)";
  
  // Note: This uses temporal scores, NOT gradients, so it won't match Integrated Gradients exactly

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '40px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>Audio Waveform</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '800px' }}>
            Temporal confidence over time. 
            <span style={{ color: accentColor, fontWeight: '700' }}> Highlighted regions</span> show {highlightMeaning}.
            <br /><span style={{ fontSize: '13px', color: '#666', fontStyle: 'italic' }}>
              Note: This shows WHERE the model was confident, while Integrated Gradients shows WHY (which features mattered).
            </span>
          </p>
        </div>

        {/* Minimal Waveform */}
        <div style={{ background: '#000', borderRadius: '16px', padding: '32px 24px', border: '1px solid rgba(255,255,255,0.08)', position: 'relative', overflow: 'hidden' }}>
          <svg width="100%" height="200" viewBox="0 0 1200 200" preserveAspectRatio="none" style={{ display: 'block' }}>
            <defs>
              <linearGradient id="cleanWave" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.6" />
                <stop offset="50%" stopColor="#8b5cf6" stopOpacity="0.6" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.6" />
              </linearGradient>
              <filter id="softGlow">
                <feGaussianBlur stdDeviation="1.5" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Centerline */}
            <line x1="0" y1="100" x2="1200" y2="100" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />

            {/* Suspicious backgrounds - subtle */}
            {scores.map((score, i) => {
              if (score <= avgScore) return null;
              const x = (timestamps[i] / maxTime) * 1200;
              const nextX = i < timestamps.length - 1 ? (timestamps[i + 1] / maxTime) * 1200 : 1200;
              return (
                <rect key={`bg-${i}`} x={x} y="0" width={nextX - x} height="200"
                  fill={`${accentColor}08`} />
              );
            })}

            {/* Main waveform - single clean line */}
            <path d={(() => {
              let p = 'M 0,100 ';
              wave.forEach((amp, i) => {
                const x = (i / samples) * 1200;
                const y = 100 + (amp * 85);
                p += `L ${x},${y} `;
              });
              return p;
            })()} fill="none" stroke="url(#cleanWave)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" filter="url(#softGlow)" />

            {/* Suspicious region overlays - bold */}
            {scores.map((score, i) => {
              if (score <= avgScore) return null;
              const si = Math.floor((timestamps[i] / maxTime) * samples);
              const ei = Math.min(i < timestamps.length - 1 ? Math.floor((timestamps[i + 1] / maxTime) * samples) : samples - 1, samples - 1);
              if (si >= ei) return null;
              let p = `M ${(si / samples) * 1200},100 `;
              for (let j = si; j <= ei; j++) {
                const x = (j / samples) * 1200;
                const y = 100 + (wave[j] * 85);
                p += `L ${x},${y} `;
              }
              return (
                <path key={`sus-${i}`} d={p} fill="none" stroke={accentColor}
                  strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" filter="url(#softGlow)" />
              );
            })}
          </svg>

          {/* Clean timeline */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '20px', paddingTop: '16px', borderTop: '1px solid rgba(255,255,255,0.06)', color: '#666', fontSize: '12px', fontWeight: '600' }}>
            <span>0.0s</span>
            <span>{maxTime.toFixed(1)}s</span>
          </div>
        </div>

        {/* Minimal stats */}
        <div style={{ marginTop: '24px', display: 'flex', gap: '16px', alignItems: 'center', padding: '20px 24px', background: 'rgba(255,255,255,0.02)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ flex: 1 }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '6px', fontWeight: '700', letterSpacing: '0.5px' }}>ANALYSIS</div>
            <div style={{ color: '#fff', fontSize: '14px', fontWeight: '600' }}>
              {highScoreRegions > 0 
                ? `${highScoreRegions} high-confidence region${highScoreRegions > 1 ? 's' : ''} (${highScorePercent}% of audio)`
                : 'Uniform confidence across audio'}
            </div>
          </div>
          {highScoreRegions > 0 && (
            <div style={{ 
              padding: '12px 20px', 
              background: `${accentColor}15`, 
              borderRadius: '8px', 
              border: `1px solid ${accentColor}30`,
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: accentColor, boxShadow: `0 0 8px ${accentColor}` }} />
              <span style={{ color: accentColor, fontSize: '13px', fontWeight: '700' }}>{isFake ? 'HIGH FAKE' : 'HIGH REAL'}</span>
            </div>
          )}
        </div>
      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};

export const ImprovedTemporalCard = ({ data, accentColor, delay, isFake }) => {
  if (!data?.timestamps) return null;
  const { timestamps, scores, mean_score, std_score, consistency_index } = data;
  const variation = std_score * 100;
  const meanPercent = mean_score * 100;
  
  // Calculate VISUAL variation (how much bars differ from each other)
  // This is more intuitive than std_score alone
  let visualVariation = 0;
  for (let i = 1; i < scores.length; i++) {
    visualVariation += Math.abs(scores[i] - scores[i-1]);
  }
  visualVariation = (visualVariation / (scores.length - 1)) * 100;
  
  // Calculate coefficient of variation (normalized std)
  const coefficientOfVariation = (std_score / (mean_score + 0.001)) * 100;
  
  // IMPROVED LOGIC based on VISUAL appearance:
  // For FAKE audio: Low variation is SUSPICIOUS (too uniform = TTS)
  // For REAL audio: Any variation is NORMAL
  let status, statusColor, statusText;
  
  if (isFake) {
    // For fake audio: check if bars look too uniform
    if (visualVariation < 3 && variation < 5) {
      status = 'bad';
      statusColor = '#ef4444';
      statusText = 'TOO UNIFORM';
    } else if (visualVariation > 8 || variation > 12) {
      status = 'good';
      statusColor = '#22c55e';
      statusText = 'NATURAL VARIATION';
    } else {
      status = 'medium';
      statusColor = '#fbbf24';
      statusText = 'MODERATE';
    }
  } else {
    // For real audio: all variation patterns are normal
    if (visualVariation < 3 && variation < 5) {
      status = 'neutral';
      statusColor = '#22c55e';
      statusText = 'CONSISTENT';
    } else if (visualVariation > 10 || variation > 15) {
      status = 'neutral';
      statusColor = '#22c55e';
      statusText = 'HIGHLY VARIED';
    } else {
      status = 'neutral';
      statusColor = '#22c55e';
      statusText = 'NORMAL VARIATION';
    }
  }

  return (
    <div style={{ opacity: 0, animation: `fadeIn 0.6s ease-out ${delay}ms forwards` }}>
      <div style={{ background: 'linear-gradient(135deg, #0a0a0a 0%, #141414 100%)', border: '2px solid #2a2a2a', borderRadius: '24px', padding: '48px', marginBottom: '32px', boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
        <div style={{ marginBottom: '40px' }}>
          <h3 style={{ color: '#fff', fontSize: '32px', fontWeight: '900', marginBottom: '12px', letterSpacing: '-0.5px' }}>Temporal Consistency Analysis</h3>
          <p style={{ color: '#888', fontSize: '15px', lineHeight: '1.7', maxWidth: '800px' }}>
            {isFake 
              ? (status === 'bad'
                ? 'Suspiciously uniform scores - too consistent across time, typical of synthetic/TTS audio'
                : status === 'good'
                ? 'Natural variation detected despite FAKE prediction - scores fluctuate naturally'
                : 'Moderate variation - some natural patterns present')
              : `${statusText} variation pattern - consistent with authentic human speech`}
          </p>
        </div>

        {/* Metrics Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '40px' }}>
          <div style={{
            background: 'rgba(255,255,255,0.02)', border: '2px solid #2a2a2a',
            borderRadius: '16px', padding: '28px', textAlign: 'center'
          }}>
            <div style={{ color: '#888', fontSize: '12px', fontWeight: '700', marginBottom: '12px', letterSpacing: '1px' }}>
              MEAN SCORE
            </div>
            <div style={{ color: '#fff', fontSize: '44px', fontWeight: '900', marginBottom: '8px' }}>
              {meanPercent.toFixed(1)}%
            </div>
            <div style={{ color: '#666', fontSize: '11px' }}>Average fakeness</div>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.02)', border: `2px solid ${statusColor}40`,
            borderRadius: '16px', padding: '28px', textAlign: 'center',
            boxShadow: `0 0 24px ${statusColor}20`
          }}>
            <div style={{ color: '#888', fontSize: '12px', fontWeight: '700', marginBottom: '12px', letterSpacing: '1px' }}>
              VARIATION
            </div>
            <div style={{ color: statusColor, fontSize: '44px', fontWeight: '900', marginBottom: '8px' }}>
              {variation.toFixed(1)}%
            </div>
            <div style={{ color: statusColor, fontSize: '11px', fontWeight: '700' }}>
              {statusText}
            </div>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.02)', border: '2px solid #2a2a2a',
            borderRadius: '16px', padding: '28px', textAlign: 'center'
          }}>
            <div style={{ color: '#888', fontSize: '12px', fontWeight: '700', marginBottom: '12px', letterSpacing: '1px' }}>
              CONSISTENCY
            </div>
            <div style={{ color: '#fff', fontSize: '44px', fontWeight: '900', marginBottom: '8px' }}>
              {(consistency_index || 0).toFixed(3)}
            </div>
            <div style={{ color: '#666', fontSize: '11px' }}>Index value</div>
          </div>
        </div>

        {/* Timeline Chart */}
        <div style={{ background: 'rgba(0,0,0,0.6)', borderRadius: '16px', padding: '40px 32px 32px 60px', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }}>
          {/* Y-axis labels */}
          <div style={{ position: 'absolute', left: '8px', top: '40px', bottom: '60px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', fontSize: '11px', color: '#666', fontWeight: '700' }}>
            <span>100%</span>
            <span>75%</span>
            <span>50%</span>
            <span>25%</span>
            <span>0%</span>
          </div>

          {/* Chart area */}
          <div style={{ position: 'relative', height: '280px' }}>
            {/* Grid lines */}
            {[0, 25, 50, 75, 100].map(pct => (
              <div key={pct} style={{
                position: 'absolute', left: 0, right: 0, bottom: `${pct}%`,
                height: '1px', background: 'rgba(255,255,255,0.06)'
              }} />
            ))}

            {/* Mean line */}
            <div style={{
              position: 'absolute', left: 0, right: 0, bottom: `${meanPercent}%`,
              height: '2px', background: '#fbbf24', opacity: 0.5,
              boxShadow: '0 0 8px rgba(251, 191, 36, 0.6)'
            }} />

            {/* Bars */}
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1px', height: '100%' }}>
              {scores.map((score, i) => {
                const height = score * 100;
                const isAboveMean = score > mean_score;
                return (
                  <div key={i} style={{
                    flex: 1, height: `${height}%`, minWidth: '3px',
                    background: isAboveMean 
                      ? `linear-gradient(180deg, ${accentColor} 0%, ${accentColor}cc 100%)`
                      : 'linear-gradient(180deg, #22c55e 0%, #16a34a 100%)',
                    borderRadius: '2px 2px 0 0',
                    boxShadow: isAboveMean ? `0 0 8px ${accentColor}66` : 'none',
                    transition: 'all 0.3s ease',
                    cursor: 'pointer'
                  }}
                  title={`${timestamps[i]?.toFixed(2)}s: ${(score * 100).toFixed(1)}%`}
                  />
                );
              })}
            </div>
          </div>

          {/* X-axis */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '16px', paddingTop: '12px', borderTop: '1px solid rgba(255,255,255,0.06)', color: '#666', fontSize: '13px', fontWeight: '700' }}>
            <span>0.0s</span>
            <span style={{ color: '#888' }}>TIME →</span>
            <span>{timestamps[timestamps.length - 1]?.toFixed(1)}s</span>
          </div>
        </div>

      </div>
      <style>{`@keyframes fadeIn { to { opacity: 1; } }`}</style>
    </div>
  );
};
