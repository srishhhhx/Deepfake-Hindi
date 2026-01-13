import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ScatterChart, Scatter, Cell, ReferenceLine } from 'recharts';

const DiffHeatmap = ({ realMatrix, fakeMatrix }) => {
  if (!Array.isArray(realMatrix) || realMatrix.length === 0 || !Array.isArray(realMatrix[0])) return null;
  if (!Array.isArray(fakeMatrix) || fakeMatrix.length === 0 || !Array.isArray(fakeMatrix[0])) return null;

  const nF = Math.min(realMatrix.length, fakeMatrix.length);
  const nT = Math.min(realMatrix[0].length, fakeMatrix[0].length);
  if (nF === 0 || nT === 0) return null;

  const absVals = [];
  for (let i = 0; i < nF; i++) {
    const rr = realMatrix[i];
    const ff = fakeMatrix[i];
    for (let j = 0; j < nT; j++) {
      const rv = rr[j];
      const fv = ff[j];
      if (typeof rv !== 'number' || typeof fv !== 'number' || !Number.isFinite(rv) || !Number.isFinite(fv)) continue;
      absVals.push(Math.abs(fv - rv));
    }
  }
  absVals.sort((a, b) => a - b);
  const p99 = absVals.length > 0 ? absVals[Math.max(0, Math.min(absVals.length - 1, Math.floor(0.99 * (absVals.length - 1))))] : 0;
  const mxAbs = Number.isFinite(p99) && p99 > 1e-12 ? p99 : 1e-6;

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

    const img = ctx.createImageData(w, h);
    const dataArr = img.data;

    const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
    const diverge = (t) => {
      const x = clamp(t, -1, 1);
      const a = Math.abs(x);
      const gamma = Math.pow(a, 0.85);
      const c0 = [32, 32, 36];
      if (x >= 0) {
        const c1 = [230, 72, 120];
        const r = Math.round(c0[0] + (c1[0] - c0[0]) * gamma);
        const g = Math.round(c0[1] + (c1[1] - c0[1]) * gamma);
        const b = Math.round(c0[2] + (c1[2] - c0[2]) * gamma);
        return [r, g, b];
      }
      const c1 = [60, 110, 245];
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * gamma);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * gamma);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * gamma);
      return [r, g, b];
    };

    const safeDiff = (f, t) => {
      const rv = realMatrix[f]?.[t];
      const fv = fakeMatrix[f]?.[t];
      if (typeof rv !== 'number' || typeof fv !== 'number' || !Number.isFinite(rv) || !Number.isFinite(fv)) return 0;
      return fv - rv;
    };

    const scale = mxAbs * 1.5;
    const denom = Math.asinh(mxAbs / scale);
    for (let y = 0; y < h; y++) {
      const fPos = ((h - 1 - y) / (h - 1)) * (nF - 1);
      const f0 = Math.floor(fPos);
      const f1 = Math.min(nF - 1, f0 + 1);
      const wf = fPos - f0;
      for (let x = 0; x < w; x++) {
        const tPos = (x / (w - 1)) * (nT - 1);
        const t0 = Math.floor(tPos);
        const t1 = Math.min(nT - 1, t0 + 1);
        const wt = tPos - t0;

        const d00 = safeDiff(f0, t0);
        const d01 = safeDiff(f0, t1);
        const d10 = safeDiff(f1, t0);
        const d11 = safeDiff(f1, t1);

        const d0 = d00 * (1 - wt) + d01 * wt;
        const d1 = d10 * (1 - wt) + d11 * wt;
        const d = d0 * (1 - wf) + d1 * wf;

        const z = d / scale;
        const norm = clamp(Math.asinh(z) / denom, -1, 1);
        const [r, g, b] = diverge(norm);

        const idx = (y * w + x) * 4;
        dataArr[idx + 0] = r;
        dataArr[idx + 1] = g;
        dataArr[idx + 2] = b;
        dataArr[idx + 3] = 255;
      }
    }

    ctx.putImageData(img, 0, 0);

    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 5; i++) {
      const yy = (h * i) / 5;
      ctx.beginPath();
      ctx.moveTo(0, yy);
      ctx.lineTo(w, yy);
      ctx.stroke();
    }
  }, [realMatrix, fakeMatrix, nF, nT, mxAbs]);

  return (
    <div style={{
      background: 'rgba(0,0,0,0.55)',
      borderRadius: '14px',
      padding: '16px',
      border: '1px solid rgba(167, 139, 250, 0.2)',
      boxShadow: 'inset 0 2px 12px rgba(0,0,0,0.35)'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div style={{ color: '#ddd', fontSize: '13px', fontWeight: '800', letterSpacing: '0.3px' }}>Difference Heatmap (Fake − Real)</div>
        <div style={{ color: '#666', fontSize: '12px', fontWeight: '700', fontFamily: 'monospace' }}>
          scale |Δ| (p99) ≈ {mxAbs.toExponential(2)}
        </div>
      </div>
      <canvas ref={canvasRef} style={{ width: '100%', borderRadius: '10px', display: 'block' }} />
      <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '12px' }}>
        <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px' }}>REAL &gt; FAKE</div>
        <div style={{
          flex: 1,
          height: '10px',
          borderRadius: '999px',
          background: 'linear-gradient(90deg, rgb(60, 110, 245) 0%, rgb(47, 75, 151) 35%, rgb(32, 32, 36) 50%, rgb(141, 54, 82) 65%, rgb(230, 72, 120) 100%)',
          border: '1px solid rgba(255,255,255,0.08)'
        }} />
        <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px' }}>FAKE &gt; REAL</div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', color: '#666', fontSize: '11px', fontWeight: '700' }}>
        <span>Start</span>
        <span>End</span>
      </div>
    </div>
  );
};

const GlobalXAI = ({ onClose }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      const candidateUrls = [
        '/global_xai_results_1000_1000_seed1/global_xai_results.json',
        '/global_xai_results_seed1/global_xai_results.json',
        '/global_xai_results/global_xai_results.json',
        '/global_xai_results_v2.json'
      ];

      try {
        let lastErr = null;
        for (const url of candidateUrls) {
          try {
            const res = await fetch(url, { cache: 'no-cache' });
            if (!res.ok) {
              lastErr = new Error(`HTTP ${res.status} for ${url}`);
              continue;
            }
            const jsonData = await res.json();
            setData(jsonData);
            setLoading(false);
            return;
          } catch (e) {
            lastErr = e;
          }
        }

        throw lastErr || new Error('Failed to load global XAI JSON');
      } catch (err) {
        const tried = candidateUrls.join(', ');
        setError(`Failed to load global XAI data (static JSON). Tried: ${tried}`);
        setLoading(false);
      }
    };
    loadData();
  }, []);

  if (loading) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <div style={{
          width: '60px',
          height: '60px',
          border: '4px solid #333',
          borderTop: '4px solid #fff',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite'
        }} />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '40px'
      }}>
        <div style={{
          background: 'rgba(239, 68, 68, 0.1)',
          border: '1px solid #ef4444',
          borderRadius: '12px',
          padding: '32px',
          maxWidth: '500px',
          color: '#fff'
        }}>
          <p style={{ fontSize: '18px', marginBottom: '12px' }}>{error}</p>
          <p style={{ color: '#999', fontSize: '14px' }}>Run the global_xai_analysis.py script first</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return <div>Loading...</div>;
  }

  const accentReal = '#22c55e';
  const accentFake = '#ef4444';

  // Utility
  const safeNum = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

  // Prepare temporal patterns data
  const temporalLen = Math.min(
    data?.temporal_patterns?.real?.avg_scores_by_position?.length || 0,
    data?.temporal_patterns?.fake?.avg_scores_by_position?.length || 0
  );

  // Prepare temporal patterns data
  const temporalData = Array.from({ length: temporalLen }, (_, idx) => ({
    position: `${idx * 2}%`,
    Real: safeNum(data.temporal_patterns.real.avg_scores_by_position[idx]),
    Fake: safeNum(data.temporal_patterns.fake.avg_scores_by_position[idx])
  }));

  // Prepare frequency band data
  const freqBands = Object.keys(data?.frequency_analysis?.band_statistics?.real || {});
  const frequencyData = freqBands.map(band => ({
    band: band.replace('Hz', ''),
    'Real Variance': safeNum(data.frequency_analysis.band_statistics.real?.[band]?.mean_variance),
    'Fake Variance': safeNum(data.frequency_analysis.band_statistics.fake?.[band]?.mean_variance)
  }));

  const frequencyMeanData = freqBands.map(band => ({
    band: band.replace('Hz', ''),
    'Real Mean': safeNum(data.frequency_analysis.band_statistics.real?.[band]?.mean_energy),
    'Fake Mean': safeNum(data.frequency_analysis.band_statistics.fake?.[band]?.mean_energy)
  }));

  const suspiciousBandsData = (Array.isArray(data?.frequency_analysis?.suspicious_bands)
    ? data.frequency_analysis.suspicious_bands
    : []).map((b) => ({
    band: (b?.freq_range || '').replace('Hz', ''),
    variance: safeNum(b?.mean_variance),
    uniformity: typeof b?.uniformity === 'number' && Number.isFinite(b.uniformity) ? b.uniformity : null
  }));

  // Frequency comparison heatmap (average spectrogram)
  const realAvgSpec = data?.frequency_analysis?.comparison_heatmap?.real_avg_spectrogram || [];
  const fakeAvgSpec = data?.frequency_analysis?.comparison_heatmap?.fake_avg_spectrogram || [];

  const hasAvgSpec = Array.isArray(realAvgSpec) && realAvgSpec.length > 0 && Array.isArray(realAvgSpec[0]);

  const Heatmap = ({ matrix, label, borderColor }) => {
    if (!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0]) || matrix[0].length === 0) return null;

    const nF = matrix.length;
    const nT = matrix[0].length;

    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < nF; i++) {
      const row = matrix[i];
      if (!Array.isArray(row)) continue;
      for (let j = 0; j < row.length; j++) {
        const v = row[j];
        if (typeof v !== 'number' || !Number.isFinite(v)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    if (!Number.isFinite(mn) || !Number.isFinite(mx) || mx <= mn) {
      mn = -80;
      mx = 0;
    }

    const canvasRef = React.useRef(null);
    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = 820;
      const h = 260;
      canvas.width = w;
      canvas.height = h;

      const img = ctx.createImageData(w, h);
      const dataArr = img.data;

      for (let y = 0; y < h; y++) {
        const fIdx = Math.floor(((h - 1 - y) / (h - 1)) * (nF - 1));
        const row = matrix[fIdx];
        for (let x = 0; x < w; x++) {
          const tIdx = Math.floor((x / (w - 1)) * (nT - 1));
          const v = Array.isArray(row) ? row[tIdx] : mn;
          const norm = Math.max(0, Math.min(1, (v - mn) / (mx - mn)));

          const r = Math.floor(20 + norm * 120);
          const g = Math.floor(20 + norm * 60);
          const b = Math.floor(45 + norm * 170);

          const idx = (y * w + x) * 4;
          dataArr[idx + 0] = r;
          dataArr[idx + 1] = g;
          dataArr[idx + 2] = b;
          dataArr[idx + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);

      ctx.strokeStyle = 'rgba(255,255,255,0.06)';
      ctx.lineWidth = 1;
      for (let i = 1; i < 5; i++) {
        const yy = (h * i) / 5;
        ctx.beginPath();
        ctx.moveTo(0, yy);
        ctx.lineTo(w, yy);
        ctx.stroke();
      }
    }, [matrix, nF, nT, mn, mx]);

    return (
      <div style={{
        background: 'rgba(0,0,0,0.55)',
        borderRadius: '14px',
        padding: '16px',
        border: `1px solid ${borderColor}`,
        boxShadow: 'inset 0 2px 12px rgba(0,0,0,0.35)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <div style={{ color: '#ddd', fontSize: '13px', fontWeight: '800', letterSpacing: '0.3px' }}>{label}</div>
          <div style={{ color: '#666', fontSize: '12px', fontWeight: '700', fontFamily: 'monospace' }}>
            dB: {mn.toFixed(0)}..{mx.toFixed(0)}
          </div>
        </div>
        <canvas ref={canvasRef} style={{ width: '100%', borderRadius: '10px', display: 'block' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', color: '#666', fontSize: '11px', fontWeight: '700' }}>
          <span>Start</span>
          <span>End</span>
        </div>
      </div>
    );
  };

  // Expert SHAP summary chart
  const shapReal = data?.expert_shap_analysis?.real || null;
  const shapFake = data?.expert_shap_analysis?.fake || null;
  const shapExperts = Object.keys(shapFake?.expert_contribution_mean || shapReal?.expert_contribution_mean || {});
  const shapBarData = shapExperts.map((name) => ({
    expert: name.includes('hubert') ? 'HuBERT' : (name.includes('wav2vec2') ? 'Wav2Vec2' : name),
    Real: safeNum(shapReal?.expert_contribution_mean?.[name]),
    Fake: safeNum(shapFake?.expert_contribution_mean?.[name])
  }));

  // Integrated Gradients summary curves
  const igReal = data?.integrated_gradients_analysis?.real || null;
  const igFake = data?.integrated_gradients_analysis?.fake || null;
  const igLen = Math.min(igReal?.combined_mean_curve?.length || 0, igFake?.combined_mean_curve?.length || 0);
  const igCurveData = Array.from({ length: igLen }, (_, i) => ({
    idx: i,
    Real: safeNum(igReal.combined_mean_curve[i]),
    Fake: safeNum(igFake.combined_mean_curve[i]),
    RealStd: safeNum(igReal.combined_std_curve?.[i]),
    FakeStd: safeNum(igFake.combined_std_curve?.[i])
  }));

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
      padding: '40px 20px',
      fontFamily: "'Montserrat', sans-serif"
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        
        {/* Back Button */}
        <button
          onClick={onClose}
          style={{
            marginBottom: '32px',
            padding: '12px 24px',
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '8px',
            color: '#fff',
            fontSize: '14px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            transition: 'all 0.3s ease'
          }}
          onMouseOver={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.1)'}
          onMouseOut={(e) => e.target.style.background = 'rgba(255, 255, 255, 0.05)'}
        >
          ← Back to Home
        </button>
        
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '48px' }}>
          <div style={{ marginBottom: '12px' }}>
            <div style={{ 
              display: 'inline-block',
              padding: '6px 16px',
              background: 'rgba(167, 139, 250, 0.1)',
              border: '1px solid rgba(167, 139, 250, 0.3)',
              borderRadius: '20px',
              fontSize: '12px',
              fontWeight: '600',
              letterSpacing: '0.5px',
              color: '#a78bfa',
              marginBottom: '20px'
            }}>
              EXPLAINABLE AI ANALYSIS
            </div>
          </div>
          <h1 style={{
            color: '#fff',
            fontSize: '52px',
            fontWeight: '700',
            marginBottom: '16px',
            letterSpacing: '-1px',
            lineHeight: '1.1'
          }}>
            Global Interpretability Report
          </h1>
          <p style={{ color: '#999', fontSize: '16px', marginBottom: '8px', lineHeight: '1.6' }}>
            Comprehensive analysis of model decision patterns across temporal, spectral, and linguistic dimensions.
          </p>
          <p style={{ color: '#666', fontSize: '14px' }}>
            Dataset: {data.summary.n_real_samples.toLocaleString()} authentic samples | {data.summary.n_fake_samples.toLocaleString()} synthetic samples
          </p>
        </div>

        {/* Summary Cards */}
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', 
          gap: '20px', 
          marginBottom: '48px' 
        }}>
          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Authentic Samples</div>
            <div style={{ color: '#fff', fontSize: '36px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-1px' }}>
              {data.summary.n_real_samples.toLocaleString()}
            </div>
            <div style={{ color: '#22c55e', fontSize: '13px', fontWeight: '500' }}>
              Baseline reference set
            </div>
          </div>

          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Synthetic Samples</div>
            <div style={{ color: '#fff', fontSize: '36px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-1px' }}>
              {data.summary.n_fake_samples.toLocaleString()}
            </div>
            <div style={{ color: '#ef4444', fontSize: '13px', fontWeight: '500' }}>
              Deepfake test set
            </div>
          </div>

          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Spectral Bands</div>
            <div style={{ color: '#fff', fontSize: '36px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-1px' }}>
              {freqBands.length}
            </div>
            <div style={{ color: '#a78bfa', fontSize: '13px', fontWeight: '500' }}>
              0-8 kHz coverage
            </div>
          </div>

          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '28px',
            boxShadow: '0 4px 16px rgba(0, 0, 0, 0.2)'
          }}>
            <div style={{ color: '#666', fontSize: '11px', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: '600' }}>Linguistic Markers</div>
            <div style={{ color: '#fff', fontSize: '36px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-1px' }}>
              {data.linguistic_patterns.high_risk_words.length}
            </div>
            <div style={{ color: '#fbbf24', fontSize: '13px', fontWeight: '500' }}>
              High-correlation lexemes
            </div>
          </div>
        </div>

        {/* Chart 1: Temporal Patterns */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '16px',
          padding: '36px',
          marginBottom: '32px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
        }}>
          <div style={{ marginBottom: '24px' }}>
            <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
              Temporal Feature Importance
            </h2>
            <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
              Average importance across time-position bins (start → end of the audio). Scores come from a perturbation-based analysis showing where changes most affect the model’s output.
            </p>
            <p style={{ color: '#666', fontSize: '13px' }}>
              Detected {data.temporal_patterns.real.hotspot_regions.length + data.temporal_patterns.fake.hotspot_regions.length} high-importance regions across both classes
            </p>
          </div>
          <ResponsiveContainer width="100%" height={500}>
            <AreaChart data={temporalData} margin={{ top: 20, right: 40, left: 10, bottom: 10 }}>
              <defs>
                <linearGradient id="colorReal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22c55e" stopOpacity={0.5}/>
                  <stop offset="50%" stopColor="#22c55e" stopOpacity={0.2}/>
                  <stop offset="100%" stopColor="#22c55e" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="colorFake" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.5}/>
                  <stop offset="50%" stopColor="#ef4444" stopOpacity={0.2}/>
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0}/>
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                  <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                  </feMerge>
                </filter>
              </defs>
              <CartesianGrid 
                strokeDasharray="1 3" 
                stroke="rgba(255,255,255,0.02)" 
                vertical={false}
                strokeWidth={1}
              />
              <XAxis 
                dataKey="position" 
                stroke="#444" 
                style={{ fontSize: '12px', fontWeight: '500', fontFamily: 'Inter, system-ui, sans-serif' }}
                tick={{ fill: '#999' }}
                axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                tickLine={{ stroke: '#444' }}
              />
              <YAxis 
                stroke="#444" 
                style={{ fontSize: '12px', fontWeight: '500', fontFamily: 'Inter, system-ui, sans-serif' }}
                tick={{ fill: '#999' }}
                axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                tickLine={{ stroke: '#444' }}
                label={{ 
                  value: 'Importance Score', 
                  angle: -90, 
                  position: 'insideLeft', 
                  style: { fill: '#888', fontSize: '13px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' } 
                }}
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
                labelStyle={{ 
                  color: '#a78bfa', 
                  fontWeight: '700', 
                  marginBottom: '8px',
                  fontSize: '14px',
                  fontFamily: 'Inter, system-ui, sans-serif'
                }}
                itemStyle={{
                  padding: '4px 0',
                  fontFamily: 'Inter, system-ui, sans-serif'
                }}
              />
              <Legend 
                wrapperStyle={{ paddingTop: '24px' }}
                iconType="line"
                iconSize={20}
                formatter={(value) => 
                  <span style={{ 
                    color: '#ddd', 
                    fontSize: '14px', 
                    fontWeight: '600',
                    fontFamily: 'Inter, system-ui, sans-serif'
                  }}>
                    {value === 'Real' ? 'Authentic Samples' : 'Synthetic Samples'}
                  </span>
                }
              />
              <Area 
                type="monotone" 
                dataKey="Real" 
                stroke="#22c55e" 
                strokeWidth={3} 
                fill="url(#colorReal)"
                dot={false}
                activeDot={{ r: 6, fill: '#22c55e', stroke: '#fff', strokeWidth: 2 }}
              />
              <Area 
                type="monotone" 
                dataKey="Fake" 
                stroke="#ef4444" 
                strokeWidth={3} 
                fill="url(#colorFake)"
                dot={false}
                activeDot={{ r: 6, fill: '#ef4444', stroke: '#fff', strokeWidth: 2 }}
              />
            </AreaChart>
          </ResponsiveContainer>
          
          {/* Hotspot Regions */}
          {(data.temporal_patterns.real.hotspot_regions.length > 0 || data.temporal_patterns.fake.hotspot_regions.length > 0) && (
            <div style={{ marginTop: '24px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <div style={{ padding: '18px', background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.08) 0%, rgba(34, 197, 94, 0.04) 100%)', border: '1px solid rgba(34, 197, 94, 0.25)', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)' }}>
                <div style={{ color: '#22c55e', fontSize: '11px', fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Authentic - Critical Regions</div>
                {data.temporal_patterns.real.hotspot_regions.slice(0, 3).map((h, i) => (
                  <div key={i} style={{ color: '#ccc', fontSize: '13px', marginBottom: '4px' }}>
                    {h.start_bin}-{h.end_bin} ({typeof h.avg_score === 'number' ? h.avg_score.toFixed(3) : 'N/A'})
                  </div>
                ))}
              </div>
              <div style={{ padding: '18px', background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(239, 68, 68, 0.04) 100%)', border: '1px solid rgba(239, 68, 68, 0.25)', borderRadius: '10px', boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)' }}>
                <div style={{ color: '#ef4444', fontSize: '11px', fontWeight: '600', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Synthetic - Critical Regions</div>
                {data.temporal_patterns.fake.hotspot_regions.slice(0, 3).map((h, i) => (
                  <div key={i} style={{ color: '#ccc', fontSize: '13px', marginBottom: '4px' }}>
                    {h.start_bin}-{h.end_bin} ({typeof h.avg_score === 'number' ? h.avg_score.toFixed(3) : 'N/A'})
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '16px',
          padding: '36px',
          marginBottom: '32px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
        }}>
          <div style={{ marginBottom: '24px' }}>
            <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
              Spectral Band Statistics
            </h2>
            <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
              Variance and mean mel-band energy across the dataset (Real vs Fake). Variance highlights stability/smoothness; mean captures global energy shifts.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px' }}>
            <div style={{
              background: 'rgba(0,0,0,0.35)',
              borderRadius: '14px',
              padding: '16px',
              border: '1px solid rgba(255,255,255,0.06)'
            }}>
              <div style={{ color: '#ddd', fontSize: '13px', fontWeight: '800', letterSpacing: '0.3px', marginBottom: '10px' }}>Variance</div>
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={frequencyData} margin={{ top: 10, right: 20, left: 0, bottom: 68 }}>
                  <defs>
                    <linearGradient id="barReal" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22c55e" stopOpacity={1}/>
                      <stop offset="100%" stopColor="#16a34a" stopOpacity={0.85}/>
                    </linearGradient>
                    <linearGradient id="barFake" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={1}/>
                      <stop offset="100%" stopColor="#dc2626" stopOpacity={0.85}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="1 3" stroke="rgba(255,255,255,0.02)" vertical={false} strokeWidth={1} />
                  <XAxis
                    dataKey="band"
                    stroke="#444"
                    angle={-45}
                    textAnchor="end"
                    height={70}
                    style={{ fontSize: '11px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' }}
                    tick={{ fill: '#999' }}
                    axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                    tickLine={{ stroke: '#444' }}
                    interval={0}
                  />
                  <YAxis
                    stroke="#444"
                    style={{ fontSize: '12px', fontWeight: '500', fontFamily: 'Inter, system-ui, sans-serif' }}
                    tick={{ fill: '#999' }}
                    axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                    tickLine={{ stroke: '#444' }}
                    label={{ value: 'Variance', angle: -90, position: 'insideLeft', style: { fill: '#888', fontSize: '13px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' } }}
                  />
                  <Tooltip
                    contentStyle={{ background: 'rgba(5, 5, 5, 0.97)', backdropFilter: 'blur(12px)', border: '1px solid rgba(167, 139, 250, 0.2)', borderRadius: '12px', color: '#fff', boxShadow: '0 8px 32px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)', fontSize: '13px', padding: '12px 16px' }}
                    labelStyle={{ color: '#a78bfa', fontWeight: '700', marginBottom: '8px', fontSize: '14px', fontFamily: 'Inter, system-ui, sans-serif' }}
                    formatter={(value, name) => [typeof value === 'number' ? value.toFixed(3) : value, name === 'Real Variance' ? 'Authentic' : 'Synthetic']}
                    itemStyle={{ padding: '4px 0', fontFamily: 'Inter, system-ui, sans-serif', fontWeight: '600' }}
                    cursor={{ fill: 'rgba(167, 139, 250, 0.05)' }}
                  />
                  <Legend verticalAlign="bottom" height={22} iconSize={12} iconType="square" formatter={(value) => <span style={{ color: '#ddd', fontSize: '12px', fontWeight: '700', fontFamily: 'Inter, system-ui, sans-serif' }}>{value === 'Real Variance' ? 'Authentic' : 'Synthetic'}</span>} />
                  <Bar dataKey="Real Variance" fill="url(#barReal)" radius={[6, 6, 0, 0]} maxBarSize={44} />
                  <Bar dataKey="Fake Variance" fill="url(#barFake)" radius={[6, 6, 0, 0]} maxBarSize={44} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div style={{
              background: 'rgba(0,0,0,0.35)',
              borderRadius: '14px',
              padding: '16px',
              border: '1px solid rgba(255,255,255,0.06)'
            }}>
              <div style={{ color: '#ddd', fontSize: '13px', fontWeight: '800', letterSpacing: '0.3px', marginBottom: '10px' }}>Mean Energy</div>
              <ResponsiveContainer width="100%" height={360}>
                <BarChart data={frequencyMeanData} margin={{ top: 10, right: 20, left: 0, bottom: 68 }}>
                  <defs>
                    <linearGradient id="meanReal" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22c55e" stopOpacity={1}/>
                      <stop offset="100%" stopColor="#16a34a" stopOpacity={0.85}/>
                    </linearGradient>
                    <linearGradient id="meanFake" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={1}/>
                      <stop offset="100%" stopColor="#dc2626" stopOpacity={0.85}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="1 3" stroke="rgba(255,255,255,0.02)" vertical={false} strokeWidth={1} />
                  <XAxis
                    dataKey="band"
                    stroke="#444"
                    angle={-45}
                    textAnchor="end"
                    height={70}
                    style={{ fontSize: '11px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' }}
                    tick={{ fill: '#999' }}
                    axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                    tickLine={{ stroke: '#444' }}
                    interval={0}
                  />
                  <YAxis
                    stroke="#444"
                    style={{ fontSize: '12px', fontWeight: '500', fontFamily: 'Inter, system-ui, sans-serif' }}
                    tick={{ fill: '#999' }}
                    axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                    tickLine={{ stroke: '#444' }}
                    tickFormatter={(v) => (typeof v === 'number' ? v.toExponential(2) : v)}
                    label={{ value: 'Mean', angle: -90, position: 'insideLeft', style: { fill: '#888', fontSize: '13px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' } }}
                  />
                  <Tooltip
                    contentStyle={{ background: 'rgba(5, 5, 5, 0.97)', backdropFilter: 'blur(12px)', border: '1px solid rgba(167, 139, 250, 0.2)', borderRadius: '12px', color: '#fff', boxShadow: '0 8px 32px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)', fontSize: '13px', padding: '12px 16px' }}
                    labelStyle={{ color: '#a78bfa', fontWeight: '700', marginBottom: '8px', fontSize: '14px', fontFamily: 'Inter, system-ui, sans-serif' }}
                    formatter={(value, name) => [typeof value === 'number' ? value.toExponential(3) : value, name === 'Real Mean' ? 'Authentic' : 'Synthetic']}
                    itemStyle={{ padding: '4px 0', fontFamily: 'Inter, system-ui, sans-serif', fontWeight: '600' }}
                    cursor={{ fill: 'rgba(167, 139, 250, 0.05)' }}
                  />
                  <Legend verticalAlign="bottom" height={22} iconSize={12} iconType="square" formatter={(value) => <span style={{ color: '#ddd', fontSize: '12px', fontWeight: '700', fontFamily: 'Inter, system-ui, sans-serif' }}>{value === 'Real Mean' ? 'Authentic' : 'Synthetic'}</span>} />
                  <Bar dataKey="Real Mean" fill="url(#meanReal)" radius={[6, 6, 0, 0]} maxBarSize={44} />
                  <Bar dataKey="Fake Mean" fill="url(#meanFake)" radius={[6, 6, 0, 0]} maxBarSize={44} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {suspiciousBandsData.length > 0 && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '36px',
            marginBottom: '32px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
          }}>
            <div style={{ marginBottom: '24px' }}>
              <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
                Suspicious Frequency Bands (Low Variance)
              </h2>
              <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
                Bands flagged as unusually uniform (low variance), often consistent with vocoder smoothing artifacts.
              </p>
            </div>
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={suspiciousBandsData} margin={{ top: 10, right: 20, left: 0, bottom: 58 }}>
                <defs>
                  <linearGradient id="susBar" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.95}/>
                    <stop offset="100%" stopColor="#7c3aed" stopOpacity={0.8}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="1 3" stroke="rgba(255,255,255,0.02)" vertical={false} strokeWidth={1} />
                <XAxis
                  dataKey="band"
                  stroke="#444"
                  angle={-30}
                  textAnchor="end"
                  height={62}
                  style={{ fontSize: '11px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' }}
                  tick={{ fill: '#999' }}
                  axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                  tickLine={{ stroke: '#444' }}
                  interval={0}
                />
                <YAxis
                  stroke="#444"
                  style={{ fontSize: '12px', fontWeight: '500', fontFamily: 'Inter, system-ui, sans-serif' }}
                  tick={{ fill: '#999' }}
                  axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                  tickLine={{ stroke: '#444' }}
                  label={{ value: 'Mean Variance', angle: -90, position: 'insideLeft', style: { fill: '#888', fontSize: '13px', fontWeight: '600', fontFamily: 'Inter, system-ui, sans-serif' } }}
                />
                <Tooltip
                  contentStyle={{ background: 'rgba(5, 5, 5, 0.97)', backdropFilter: 'blur(12px)', border: '1px solid rgba(167, 139, 250, 0.2)', borderRadius: '12px', color: '#fff', boxShadow: '0 8px 32px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)', fontSize: '13px', padding: '12px 16px' }}
                  labelStyle={{ color: '#a78bfa', fontWeight: '700', marginBottom: '8px', fontSize: '14px', fontFamily: 'Inter, system-ui, sans-serif' }}
                  formatter={(value, name, item) => {
                    if (name !== 'variance') return [value, name];
                    const u = item?.payload?.uniformity;
                    const extra = (typeof u === 'number') ? ` (uniformity ${u.toFixed(3)})` : '';
                    return [typeof value === 'number' ? value.toFixed(6) : value, `Mean Variance${extra}`];
                  }}
                  itemStyle={{ padding: '4px 0', fontFamily: 'Inter, system-ui, sans-serif', fontWeight: '600' }}
                  cursor={{ fill: 'rgba(167, 139, 250, 0.05)' }}
                />
                <Bar dataKey="variance" fill="url(#susBar)" radius={[6, 6, 0, 0]} maxBarSize={56} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* High-Risk Words */}
        <div style={{
          background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          borderRadius: '16px',
          padding: '36px',
          marginBottom: '32px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
        }}>
          <div style={{ marginBottom: '32px' }}>
            <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
              Linguistic Correlation Analysis
            </h2>
            <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
              Lexical items exhibiting statistically significant correlation with elevated model confidence scores for synthetic classification. These patterns may reflect phonetic complexity or pronunciation artifacts in TTS systems.
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginTop: '12px' }}>
              <div style={{ 
                padding: '6px 14px', 
                background: 'rgba(251, 191, 36, 0.1)', 
                border: '1px solid rgba(251, 191, 36, 0.3)',
                borderRadius: '20px',
                fontSize: '12px',
                fontWeight: '600',
                color: '#fbbf24'
              }}>
                {data.linguistic_patterns.high_risk_words.length} Markers Detected
              </div>
            </div>
          </div>
          
          {/* Table Header */}
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: '60px 1fr 140px 120px 120px',
            gap: '16px',
            padding: '16px 20px',
            background: 'rgba(255, 255, 255, 0.02)',
            borderRadius: '10px 10px 0 0',
            borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
            marginBottom: '0'
          }}>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Rank</div>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Lexeme</div>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Confidence</div>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Frequency</div>
            <div style={{ color: '#888', fontSize: '11px', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Impact</div>
          </div>
          
          {/* Table Rows */}
          <div style={{ 
            background: 'rgba(255, 255, 255, 0.01)',
            borderRadius: '0 0 10px 10px',
            overflow: 'hidden'
          }}>
            {data.linguistic_patterns.high_risk_words.slice(0, 10).map((word, idx) => {
              const score = typeof word.avg_score_when_present === 'number' ? word.avg_score_when_present : 0;
              const confidence = score * 100;
              const impactLevel = score > 0.8 ? 'Critical' : score > 0.6 ? 'High' : 'Moderate';
              const impactColor = score > 0.8 ? '#ef4444' : score > 0.6 ? '#f59e0b' : '#fbbf24';
              
              return (
                <div 
                  key={idx} 
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '60px 1fr 140px 120px 120px',
                    gap: '16px',
                    padding: '20px',
                    borderBottom: idx < 9 ? '1px solid rgba(255, 255, 255, 0.03)' : 'none',
                    transition: 'all 0.2s ease',
                    cursor: 'default'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(251, 191, 36, 0.03)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  {/* Rank Badge */}
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <div style={{
                      width: '32px',
                      height: '32px',
                      borderRadius: '8px',
                      background: idx < 3 ? `linear-gradient(135deg, ${impactColor}40, ${impactColor}20)` : 'rgba(255, 255, 255, 0.05)',
                      border: `1px solid ${idx < 3 ? impactColor + '40' : 'rgba(255, 255, 255, 0.08)'}`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '14px',
                      fontWeight: '700',
                      color: idx < 3 ? impactColor : '#999'
                    }}>
                      {idx + 1}
                    </div>
                  </div>
                  
                  {/* Word */}
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ 
                      fontSize: '18px', 
                      fontWeight: '600', 
                      color: '#fff',
                      fontFamily: 'Noto Sans Devanagari, sans-serif'
                    }}>
                      {word.word}
                    </span>
                  </div>
                  
                  {/* Confidence Bar */}
                  <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <div style={{ 
                      height: '6px', 
                      background: 'rgba(255, 255, 255, 0.05)', 
                      borderRadius: '3px',
                      overflow: 'hidden',
                      marginBottom: '6px'
                    }}>
                      <div style={{
                        height: '100%',
                        width: `${confidence}%`,
                        background: `linear-gradient(90deg, ${impactColor}, ${impactColor}cc)`,
                        borderRadius: '3px',
                        transition: 'width 0.3s ease'
                      }} />
                    </div>
                    <span style={{ fontSize: '12px', fontWeight: '600', color: '#ccc' }}>
                      {score.toFixed(3)}
                    </span>
                  </div>
                  
                  {/* Frequency */}
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <span style={{ fontSize: '15px', fontWeight: '600', color: '#aaa' }}>
                      {word.occurrences}×
                    </span>
                  </div>
                  
                  {/* Impact Badge */}
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <div style={{
                      padding: '6px 12px',
                      background: `${impactColor}15`,
                      border: `1px solid ${impactColor}40`,
                      borderRadius: '6px',
                      fontSize: '11px',
                      fontWeight: '700',
                      color: impactColor,
                      textTransform: 'uppercase',
                      letterSpacing: '0.5px'
                    }}>
                      {impactLevel}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Expert SHAP (Global) */}
        {shapBarData.length > 0 && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '36px',
            marginBottom: '32px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
          }}>
            <div style={{ marginBottom: '24px' }}>
              <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
                Expert SHAP Summary
              </h2>
              <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
                Average Shapley contribution per expert across the evaluation set.
              </p>
            </div>
            {(shapReal && shapFake) && (
              <div style={{
                marginBottom: '16px',
                padding: '14px 16px',
                background: 'rgba(255,255,255,0.02)',
                borderRadius: '12px',
                border: '1px solid rgba(255,255,255,0.06)',
                color: '#aaa',
                fontSize: '13px',
                fontFamily: 'monospace',
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '10px'
              }}>
                <div>
                  <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px' }}>REAL</div>
                  <div>baseline P(fake)={shapReal.baseline_pred_mean.toFixed(3)}</div>
                  <div>actual P(fake)={shapReal.actual_pred_mean.toFixed(3)}</div>
                </div>
                <div>
                  <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px' }}>FAKE</div>
                  <div>baseline P(fake)={shapFake.baseline_pred_mean.toFixed(3)}</div>
                  <div>actual P(fake)={shapFake.actual_pred_mean.toFixed(3)}</div>
                </div>
              </div>
            )}
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={shapBarData} margin={{ top: 20, right: 40, left: 10, bottom: 30 }}>
                <CartesianGrid strokeDasharray="1 3" stroke="rgba(255,255,255,0.02)" vertical={false} />
                <XAxis dataKey="expert" stroke="#444" tick={{ fill: '#999' }} axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }} tickLine={{ stroke: '#444' }} />
                <YAxis stroke="#444" tick={{ fill: '#999' }} axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }} tickLine={{ stroke: '#444' }} />
                <ReferenceLine y={0} stroke="rgba(255,255,255,0.12)" strokeWidth={2} />
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
                  itemStyle={{ padding: '4px 0', fontWeight: '600' }}
                  cursor={{ fill: 'rgba(167, 139, 250, 0.06)' }}
                />
                <Legend wrapperStyle={{ paddingTop: '18px' }} />
                <Bar dataKey="Real" fill={accentReal} radius={[6, 6, 0, 0]} maxBarSize={90} />
                <Bar dataKey="Fake" fill={accentFake} radius={[6, 6, 0, 0]} maxBarSize={90} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Integrated Gradients (Global) */}
        {igCurveData.length > 0 && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '36px',
            marginBottom: '32px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
          }}>
            <div style={{ marginBottom: '24px' }}>
              <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
                Integrated Gradients (Aggregate)
              </h2>
              <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
                Mean integrated gradients curve across samples. Bins 0-24 correspond to <strong>mean</strong>-derived features; bins 25-49 correspond to <strong>std</strong>-derived features.
              </p>
            </div>
            <ResponsiveContainer width="100%" height={460}>
              <LineChart data={igCurveData} margin={{ top: 20, right: 40, left: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="1 3" stroke="rgba(255,255,255,0.02)" vertical={false} />
                <ReferenceLine x={24.5} stroke="rgba(255,255,255,0.10)" strokeDasharray="3 4" />
                <XAxis
                  dataKey="idx"
                  stroke="#444"
                  tick={{ fill: '#999' }}
                  axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }}
                  tickLine={{ stroke: '#444' }}
                  ticks={[0, 24, 25, 49].filter(t => t >= 0 && t < igLen)}
                  tickFormatter={(v) => {
                    if (v === 0) return 'μ0';
                    if (v === 24) return 'μ24';
                    if (v === 25) return 'σ0';
                    if (v === 49) return 'σ24';
                    return String(v);
                  }}
                />
                <YAxis stroke="#444" tick={{ fill: '#999' }} axisLine={{ stroke: '#2a2a2a', strokeWidth: 1.5 }} tickLine={{ stroke: '#444' }} />
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
                  cursor={{ stroke: 'rgba(167, 139, 250, 0.28)', strokeWidth: 1 }}
                />
                <Legend wrapperStyle={{ paddingTop: '18px' }} />
                <Line type="monotone" dataKey="Real" stroke={accentReal} strokeWidth={3} dot={false} activeDot={{ r: 5, fill: accentReal, stroke: '#fff', strokeWidth: 2 }} />
                <Line type="monotone" dataKey="Fake" stroke={accentFake} strokeWidth={3} dot={false} activeDot={{ r: 5, fill: accentFake, stroke: '#fff', strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
            <div style={{ marginTop: '16px', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px' }}>
              {typeof igReal?.second_half_to_first_half_ratio_mean === 'number' && (
                <div style={{ padding: '16px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px' }}>REAL RATIO</div>
                  <div style={{ color: '#fff', fontSize: '20px', fontWeight: '800' }}>{igReal.second_half_to_first_half_ratio_mean.toFixed(2)}</div>
                </div>
              )}
              {typeof igFake?.second_half_to_first_half_ratio_mean === 'number' && (
                <div style={{ padding: '16px', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)' }}>
                  <div style={{ color: '#666', fontSize: '11px', fontWeight: '800', letterSpacing: '0.5px' }}>FAKE RATIO</div>
                  <div style={{ color: '#fff', fontSize: '20px', fontWeight: '800' }}>{igFake.second_half_to_first_half_ratio_mean.toFixed(2)}</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Average Spectrogram Comparison (Global) */}
        {hasAvgSpec && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.04) 0%, rgba(255, 255, 255, 0.02) 100%)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '16px',
            padding: '36px',
            marginBottom: '32px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.25)'
          }}>
            <div style={{ marginBottom: '24px' }}>
              <h2 style={{ color: '#fff', fontSize: '26px', fontWeight: '600', marginBottom: '8px', letterSpacing: '-0.5px' }}>
                Spectrogram Difference Map
              </h2>
              <p style={{ color: '#888', fontSize: '14px', lineHeight: '1.6', marginBottom: '4px' }}>
                Single view of spectral differences: values show where synthetic speech has systematically higher/lower energy than authentic speech.
              </p>
            </div>
            <DiffHeatmap realMatrix={realAvgSpec} fakeMatrix={fakeAvgSpec} />
          </div>
        )}

      </div>
    </div>
  );
};

export default GlobalXAI;
