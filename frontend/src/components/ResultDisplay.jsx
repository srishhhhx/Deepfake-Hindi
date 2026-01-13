import { useState } from 'react';
import XAIVisualizations from './XAIComponents_Final';

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const ResultDisplay = ({ result, fileName, fileSize, onReset }) => {
  const [xaiData, setXaiData] = useState(null);
  const [xaiLoading, setXaiLoading] = useState(false);
  const [xaiError, setXaiError] = useState('');
  const [exportError, setExportError] = useState('');
  console.log('ResultDisplay received:', { result, fileName, fileSize });
  
  // Early validation
  if (!result || typeof result !== 'object') {
    console.error('Invalid result:', result);
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '80px 40px',
        color: '#ef4444',
        fontSize: '18px',
        fontFamily: "'Montserrat', sans-serif"
      }}>
        <p>Error: No data received from server</p>
        <button
          onClick={onReset}
          style={{
            marginTop: '24px',
            padding: '12px 24px',
            background: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: '8px',
            color: '#fff',
            cursor: 'pointer',
            fontFamily: "'Montserrat', sans-serif"
          }}
        >
          Try Again
        </button>
      </div>
    );
  }

  const formatSize = (bytes) => {
    if (!bytes) return '—';
    const mb = bytes / (1024 * 1024);
    return mb < 1 ? Math.round(bytes / 1024) + "KB" : mb.toFixed(1) + "MB";
  };

  const handleExportXAI = async () => {
    if (!fileName) return;
    setExportError('');
    try {
      const file = new File([new Blob()], fileName, { type: 'audio/wav' });
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch(`${API}/xai/export`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const blob = await r.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${fileName || 'audio' }_xai_plots.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (e) {
      setExportError(e.message || 'Export failed');
    }
  };
  
  const truncateFilename = (name, maxLength = 30) => {
    if (!name || name.length <= maxLength) return name || 'audio.wav';
    const ext = name.split('.').pop();
    const nameWithoutExt = name.substring(0, name.lastIndexOf('.'));
    const truncated = nameWithoutExt.substring(0, maxLength - ext.length - 3);
    return `${truncated}...${ext}`;
  };

  const labelText = (lbl) => (lbl === 1 ? "FAKE" : "REAL");

  const handleGenerateXAI = async () => {
    if (!fileName) return;
    setXaiLoading(true);
    setXaiError('');
    try {
      const file = new File([new Blob()], fileName, { type: 'audio/wav' });
      const fd = new FormData();
      fd.append('file', file);
      const r = await fetch(`${API}/xai`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const json = await r.json();
      if (json.error) {
        setXaiError(json.error);
      } else {
        setXaiData(json);
      }
    } catch (e) {
      setXaiError(e.message);
    } finally {
      setXaiLoading(false);
    }
  };

  // Check if this is a language gate rejection
  if (result.error_code === "not_hindi") {
    return (
      <div style={{ 
        width: '100%', 
        minHeight: '100vh',
        background: 'transparent',
        padding: '60px 40px',
        fontFamily: "'Montserrat', sans-serif"
      }}>
        <div style={{ maxWidth: '900px', margin: '0 auto' }}>
          
          {/* Rejection Card */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)',
            border: '2px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '24px',
            padding: '60px',
            marginBottom: '40px',
            textAlign: 'center'
          }}>
            {/* Error Icon */}
            <div style={{
              width: '80px',
              height: '80px',
              background: 'rgba(239, 68, 68, 0.2)',
              borderRadius: '50%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              margin: '0 auto 24px'
            }}>
              <svg width="40" height="40" fill="#ef4444" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>

            <h2 style={{ 
              color: '#ef4444', 
              fontSize: '32px', 
              fontWeight: '700', 
              marginBottom: '16px'
            }}>
              Language Check Failed
            </h2>
            <p style={{ 
              color: '#bbb', 
              fontSize: '16px', 
              marginBottom: '32px', 
              lineHeight: '1.6'
            }}>
              {result.message || 'Audio is not in Hindi. Please upload a Hindi audio file.'}
            </p>

            {/* Language Details */}
            <div style={{
              background: 'rgba(0, 0, 0, 0.3)',
              border: '1px solid #2a2a2a',
              borderRadius: '16px',
              padding: '24px',
              marginBottom: '32px'
            }}>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', textAlign: 'center' }}>
                <div>
                  <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Detected Language</div>
                  <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>
                    {result.lid_debug?.detected_lang?.toUpperCase() || '—'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Hindi Probability</div>
                  <div style={{ color: '#ef4444', fontSize: '24px', fontWeight: '700' }}>
                    {result.lid_debug?.p_hi ? (result.lid_debug.p_hi * 100).toFixed(1) + '%' : '0%'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Speech Content</div>
                  <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>
                    {result.lid_debug?.speech_fraction ? (result.lid_debug.speech_fraction * 100).toFixed(0) + '%' : '—'}
                  </div>
                </div>
              </div>
            </div>

            {/* File Info */}
            <div style={{ color: '#666', fontSize: '14px', marginBottom: '32px' }}>
              File: {truncateFilename(fileName) || 'audio.wav'} • {fileSize ? formatSize(fileSize) : '—'}
            </div>

            {/* Action Button */}
            <button
              onClick={onReset}
              className="shimmer-btn"
              style={{
                background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
                border: '1px solid #333',
                borderRadius: '16px',
                padding: '16px 40px',
                color: 'white',
                fontSize: '16px',
                fontWeight: '700',
                cursor: 'pointer',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <div className="shimmer" />
              <span style={{ position: 'relative', zIndex: 1 }}>Try Another File</span>
            </button>
          </div>

          {/* Technical Details (Collapsible) */}
          {result.lid_debug && (
            <details style={{ marginTop: '20px' }}>
              <summary style={{
                color: '#888',
                fontSize: '16px',
                fontWeight: '600',
                cursor: 'pointer',
                padding: '16px',
                background: 'rgba(255, 255, 255, 0.02)',
                borderRadius: '12px',
                border: '1px solid #2a2a2a'
              }}>
                Technical Details
              </summary>
              <div style={{
                marginTop: '16px',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
                border: '1px solid #2a2a2a',
                borderRadius: '16px',
                padding: '24px',
                fontSize: '14px',
                fontFamily: 'monospace',
                color: '#888'
              }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {JSON.stringify(result.lid_debug, null, 2)}
                </pre>
              </div>
            </details>
          )}
        </div>
      </div>
    );
  }

  // Normal inference result (Hindi audio passed)
  // Safety check: ensure we have the required fields
  if (result.prob_fake === undefined || result.label === undefined) {
    console.error('Invalid result format:', result);
    return (
      <div style={{ 
        textAlign: 'center', 
        padding: '80px 40px',
        color: '#ef4444',
        fontSize: '18px',
        fontFamily: "'Montserrat', sans-serif"
      }}>
        <p>Error: Invalid response from server</p>
        <button
          onClick={onReset}
          style={{
            marginTop: '24px',
            padding: '12px 24px',
            background: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: '8px',
            color: '#fff',
            cursor: 'pointer',
            fontFamily: "'Montserrat', sans-serif"
          }}
        >
          Try Again
        </button>
      </div>
    );
  }

  const isFake = result.label === 1;
  const confidence = result.prob_fake !== undefined ? (Number(result.prob_fake) * 100).toFixed(2) : '0.00';
  const uncertainty = (typeof result.uncertainty === 'number') ? result.uncertainty : null;
  const uncertaintyLevel = (typeof result.uncertainty_level === 'string') ? result.uncertainty_level : null;

  return (
    <div style={{ 
      width: '100%', 
      minHeight: '100vh',
      background: 'transparent',
      padding: '60px 40px',
      fontFamily: "'Montserrat', sans-serif"
    }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* Hero Result Card */}
        <div style={{
          background: isFake 
            ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%)'
            : 'linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%)',
          border: `2px solid ${isFake ? 'rgba(239, 68, 68, 0.3)' : 'rgba(34, 197, 94, 0.3)'}`,
          borderRadius: '32px',
          padding: '100px 60px',
          position: 'relative',
          overflow: 'hidden',
          minHeight: '500px'
        }}>
          {/* Background Pattern */}
          <div style={{
            position: 'absolute',
            top: 0,
            right: 0,
            width: '400px',
            height: '400px',
            background: isFake 
              ? 'radial-gradient(circle, rgba(239, 68, 68, 0.15) 0%, transparent 70%)'
              : 'radial-gradient(circle, rgba(34, 197, 94, 0.15) 0%, transparent 70%)',
            pointerEvents: 'none'
          }} />

          <div style={{ position: 'relative', zIndex: 1 }}>
            {/* Status Badge */}
            <div style={{ 
              display: 'inline-flex',
              alignItems: 'center',
              gap: '12px',
              background: isFake ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
              border: `1px solid ${isFake ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)'}`,
              borderRadius: '50px',
              padding: '12px 28px',
              marginBottom: '40px'
            }}>
              <div style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: isFake ? '#ef4444' : '#22c55e',
                boxShadow: `0 0 20px ${isFake ? '#ef4444' : '#22c55e'}`
              }} />
              <span style={{ 
                color: isFake ? '#ef4444' : '#22c55e',
                fontSize: '16px',
                fontWeight: '700',
                letterSpacing: '1px'
              }}>
                {isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC AUDIO'}
              </span>
            </div>

            {/* Main Result */}
            <div style={{ 
              fontSize: '120px', 
              fontWeight: '900',
              color: isFake ? '#ef4444' : '#22c55e',
              lineHeight: '1',
              marginBottom: '30px',
              letterSpacing: '-2px'
            }}>
              {labelText(result.label)}
            </div>

          </div>
        </div>

        {/* Uncertainty warnings from backend */}
        {uncertaintyLevel === 'medium' && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.06) 100%)',
            border: '1px solid rgba(245, 158, 11, 0.35)',
            borderRadius: '24px',
            padding: '24px',
            marginTop: '24px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="#f59e0b" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
              </svg>
              <h4 style={{ color: '#f59e0b', fontSize: '18px', fontWeight: 800, margin: 0 }}>Moderate Uncertainty</h4>
            </div>
            <p style={{ color: '#bfbfbf', margin: 0, lineHeight: 1.6 }}>
              Note: this prediction is moderately close to the decision boundary. Treat this result with some caution.
              {typeof uncertainty === 'number' && (
                <span style={{ color: '#d4d4d4' }}> (uncertainty {(uncertainty*100).toFixed(0)}%)</span>
              )}
            </p>
          </div>
        )}

        {uncertaintyLevel === 'high' && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(239, 68, 68, 0.06) 100%)',
            border: '1px solid rgba(239, 68, 68, 0.35)',
            borderRadius: '24px',
            padding: '24px',
            marginTop: '24px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <svg width="22" height="22" viewBox="0 0 24 24" fill="#ef4444" xmlns="http://www.w3.org/2000/svg">
                <path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>
              </svg>
              <h4 style={{ color: '#ef4444', fontSize: '18px', fontWeight: 800, margin: 0 }}>High Uncertainty</h4>
            </div>
            <p style={{ color: '#bfbfbf', margin: 0, lineHeight: 1.6 }}>
              Warning: this prediction is very close to the model’s decision boundary. In our tests, this region has a higher error rate. Do not rely on this result alone.
              {typeof uncertainty === 'number' && (
                <span style={{ color: '#d4d4d4' }}> (uncertainty {(uncertainty*100).toFixed(0)}%)</span>
              )}
            </p>
          </div>
        )}

        {/* Action Button */}
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: '40px' }}>
          <button
            onClick={onReset}
            className="shimmer-btn"
            style={{
              width: '60%',
              maxWidth: '600px',
              background: 'linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%)',
              border: '1px solid #2a2a2a',
              borderRadius: '16px',
              padding: '20px',
              color: 'white',
              fontSize: '18px',
              fontWeight: '700',
              cursor: 'pointer',
              position: 'relative',
              overflow: 'hidden',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
              transition: 'all 0.3s ease'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'translateY(-2px)';
              e.currentTarget.style.boxShadow = '0 12px 40px rgba(0, 0, 0, 0.8)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'translateY(0)';
              e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.6)';
            }}
          >
            <div className="shimmer" />
            <span style={{ position: 'relative', zIndex: 1 }}>Analyze New File</span>
          </button>
        </div>

        {/* Language Check Info */}
        {result.language_check && (
          <div style={{
            background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.05) 100%)',
            border: '1px solid rgba(34, 197, 94, 0.3)',
            borderRadius: '24px',
            padding: '32px',
            marginTop: '40px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
              <svg width="24" height="24" fill="#22c55e" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <h3 style={{ color: '#22c55e', fontSize: '20px', fontWeight: '700', margin: 0 }}>Language Verified: Hindi</h3>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>Detected Language</div>
                <div style={{ color: '#fff', fontSize: '18px', fontWeight: '600' }}>
                  {result.language_check.detected_lang?.toUpperCase() || '—'}
                </div>
              </div>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>Hindi Confidence</div>
                <div style={{ color: '#22c55e', fontSize: '18px', fontWeight: '600' }}>
                  {result.language_check.p_hi ? (result.language_check.p_hi * 100).toFixed(1) + '%' : '—'}
                </div>
              </div>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>Speech Content</div>
                <div style={{ color: '#fff', fontSize: '18px', fontWeight: '600' }}>
                  {result.language_check.speech_fraction ? (result.language_check.speech_fraction * 100).toFixed(0) + '%' : '—'}
                </div>
              </div>
              <div>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>LID Time</div>
                <div style={{ color: '#fff', fontSize: '18px', fontWeight: '600' }}>
                  {result.language_check.t_lid_ms || result.debug?.t_lid_ms || '—'} <span style={{ fontSize: '12px', color: '#666' }}>ms</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Performance Metrics */}
        <div style={{
          background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
          border: '1px solid #2a2a2a',
          borderRadius: '24px',
          padding: '40px',
          marginTop: '40px'
        }}>
          <h3 style={{ color: '#fff', fontSize: '24px', fontWeight: '700', marginBottom: '32px' }}>Performance Metrics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '24px', marginBottom: '32px' }}>
            <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', border: '1px solid #2a2a2a' }}>
              <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Language Check</div>
              <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>{result.debug?.t_lid_ms || '—'} <span style={{ fontSize: '14px', color: '#666' }}>ms</span></div>
            </div>
            <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', border: '1px solid #2a2a2a' }}>
              <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>HuBERT Time</div>
              <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>{result.debug?.t_hubert_ms || '—'} <span style={{ fontSize: '14px', color: '#666' }}>ms</span></div>
            </div>
            <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', border: '1px solid #2a2a2a' }}>
              <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>Wav2Vec2 Time</div>
              <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>{result.debug?.t_w2v_ms || '—'} <span style={{ fontSize: '14px', color: '#666' }}>ms</span></div>
            </div>
            <div style={{ padding: '20px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '12px', border: '1px solid #2a2a2a' }}>
              <div style={{ color: '#666', fontSize: '12px', marginBottom: '8px' }}>MoE Time</div>
              <div style={{ color: '#fff', fontSize: '24px', fontWeight: '700' }}>{result.debug?.t_moe_ms || '—'} <span style={{ fontSize: '14px', color: '#666' }}>ms</span></div>
            </div>
          </div>

          {/* Gate Weights */}
          {result.gate && Object.keys(result.gate).length > 0 && (
            <div style={{ marginTop: '24px' }}>
              <h4 style={{ color: '#888', fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Gate Weights</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px' }}>
                {Object.entries(result.gate).map(([k,v]) => (
                  <div key={k} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between',
                    padding: '12px 16px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: '8px'
                  }}>
                    <span style={{ color: '#888', fontSize: '14px' }}>{k}</span>
                    <span style={{ color: '#fff', fontSize: '14px', fontWeight: '600', fontFamily: 'monospace' }}>{Number(v).toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Audio File Details */}
          <div style={{ marginTop: '32px', paddingTop: '32px', borderTop: '1px solid #2a2a2a' }}>
            <h4 style={{ color: '#888', fontSize: '16px', fontWeight: '600', marginBottom: '16px' }}>Audio File Details</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
              <div style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>File Name</div>
                <div style={{ color: '#fff', fontSize: '14px', fontWeight: '600', wordBreak: 'break-all' }}>{truncateFilename(fileName)}</div>
              </div>
              <div style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>File Size</div>
                <div style={{ color: '#fff', fontSize: '14px', fontWeight: '600' }}>{formatSize(fileSize)}</div>
              </div>
              <div style={{ padding: '16px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px' }}>
                <div style={{ color: '#666', fontSize: '12px', marginBottom: '6px' }}>Duration</div>
                <div style={{ color: '#fff', fontSize: '14px', fontWeight: '600' }}>{result.debug?.audio_sec}s</div>
              </div>
            </div>
          </div>

        </div>

        {/* XAI BUTTON */}
        {!xaiData && !xaiLoading && (
          <div style={{
            textAlign: 'center',
            marginTop: '60px'
          }}>
            <button
              onClick={handleGenerateXAI}
              style={{
                padding: '24px 48px',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%)',
                border: '1px solid #2a2a2a',
                borderRadius: '16px',
                color: 'white',
                fontSize: '20px',
                fontWeight: '700',
                cursor: 'pointer',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
                transition: 'all 0.3s ease',
                letterSpacing: '0.5px'
              }}
              onMouseOver={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 12px 40px rgba(0, 0, 0, 0.8)';
              }}
              onMouseOut={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.6)';
              }}
            >
              Generate Explainability Analysis
            </button>
            <p style={{ color: '#666', fontSize: '14px', marginTop: '16px' }}>
              Detailed AI decision explanations, temporal analysis, and visual breakdowns
            </p>
          </div>
        )}

        {/* XAI LOADING */}
        {xaiLoading && (
          <div style={{
            textAlign: 'center',
            padding: '80px 40px',
            marginTop: '40px'
          }}>
            <div style={{
              width: '80px',
              height: '80px',
              border: '6px solid #333',
              borderTop: `6px solid ${isFake ? '#ef4444' : '#22c55e'}`,
              borderRadius: '50%',
              margin: '0 auto 24px',
              animation: 'spin 1s linear infinite'
            }} />
            <p style={{ color: '#888', fontSize: '18px', fontWeight: '600' }}>Generating XAI Analysis...</p>
            <p style={{ color: '#666', fontSize: '14px', marginTop: '8px' }}>This may take 10-30 seconds</p>
          </div>
        )}

        {/* XAI ERROR */}
        {xaiError && (
          <div style={{
            marginTop: '40px',
            padding: '24px',
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '16px',
            color: '#ef4444',
            textAlign: 'center'
          }}>
            <p style={{ fontSize: '16px', fontWeight: '600' }}>XAI Error: {xaiError}</p>
            <button
              onClick={handleGenerateXAI}
              style={{
                marginTop: '16px',
                padding: '12px 24px',
                background: '#ef4444',
                border: 'none',
                borderRadius: '8px',
                color: 'white',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Retry
            </button>
          </div>
        )}

        {/* XAI VISUALIZATIONS */}
        {xaiData && (
          <XAIVisualizations 
            xaiData={xaiData} 
            isFake={isFake}
            fileName={fileName}
          />
        )}

        {/* XAI EXPORT BUTTON (only after XAI is available) */}
        {xaiData && !xaiLoading && (
          <div style={{
            textAlign: 'center',
            marginTop: '48px',
            marginBottom: '8px'
          }}>
            <button
              onClick={handleExportXAI}
              style={{
                padding: '24px 48px',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%)',
                border: '1px solid #2a2a2a',
                borderRadius: '16px',
                color: 'white',
                fontSize: '20px',
                fontWeight: '700',
                cursor: 'pointer',
                boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
                transition: 'all 0.3s ease',
                letterSpacing: '0.5px'
              }}
              onMouseOver={(e) => {
                e.target.style.transform = 'translateY(-2px)';
                e.target.style.boxShadow = '0 12px 40px rgba(0, 0, 0, 0.8)';
              }}
              onMouseOut={(e) => {
                e.target.style.transform = 'translateY(0)';
                e.target.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.6)';
              }}
            >
              Download XAI Figures (Python)
            </button>
            {exportError && (
              <p style={{ color: '#f97316', fontSize: '12px', marginTop: '12px' }}>
                Export error: {exportError}
              </p>
            )}
          </div>
        )}

      </div>
      {/* End of main content area */}
    </div>
  );
};

export default ResultDisplay;
