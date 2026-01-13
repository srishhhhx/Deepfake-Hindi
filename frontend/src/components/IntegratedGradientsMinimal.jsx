import React from 'react';

/**
 * Premium Minimal Integrated Gradients Visualization
 * Shows temporal attribution - which time segments were important for the prediction
 */
export const IntegratedGradientsMinimal = ({ data, accentColor, delay, audioDuration, isFake, temporalData }) => {
  // Use temporal_heatmap for TRUE temporal analysis
  if (!temporalData?.scores || !temporalData?.timestamps) {
    return null;
  }
  
  const scores = temporalData.scores;
  const timestamps = temporalData.timestamps;
  
  if (!Array.isArray(scores) || scores.length === 0) {
    return null;
  }

  // Normalize scores to 0-1 range
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);
  const normalized = scores.map(s => (s - minScore) / (maxScore - minScore || 1));
  
  // Get actual duration from timestamps
  const duration = timestamps[timestamps.length - 1] || audioDuration || 5.0;
  
  // Find peaks (top 20% most important segments)
  const threshold = 0.7;
  const peaks = normalized.filter(n => n >= threshold).length;

  return (
    <div style={{ 
      opacity: 0, 
      animation: `fadeIn 0.8s ease-out ${delay}ms forwards`,
      marginBottom: '48px'
    }}>
      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <h3 style={{ 
          color: '#fff', 
          fontSize: '22px', 
          fontWeight: '700',
          letterSpacing: '-0.5px',
          marginBottom: '12px'
        }}>
          Temporal Analysis
        </h3>
        <p style={{ 
          color: '#888', 
          fontSize: '13px', 
          lineHeight: '1.6',
          maxWidth: '900px'
        }}>
          Tracks model confidence across <span style={{ color: accentColor, fontWeight: '600' }}>time segments</span> by re-extracting features for each audio window.
          The line shows confidence flow, with peaks indicating segments where the model was most certain in its {isFake ? 'FAKE' : 'REAL'} prediction.
        </p>
      </div>

      {/* Visualization Container */}
      <div style={{
        background: 'linear-gradient(135deg, rgba(15, 15, 25, 0.6) 0%, rgba(5, 5, 15, 0.8) 100%)',
        borderRadius: '24px',
        padding: '48px 40px 36px',
        border: '1px solid rgba(100, 150, 255, 0.15)',
        position: 'relative',
        boxShadow: '0 4px 20px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(100, 150, 255, 0.1)'
      }}>
        {/* Line Chart Visualization */}
        <svg
          width="100%"
          height="240"
          style={{ marginBottom: '20px' }}
          viewBox="0 0 1000 240"
          preserveAspectRatio="none"
        >
          {/* Grid lines */}
          {[0, 60, 120, 180, 240].map((y, i) => (
            <line
              key={i}
              x1="0"
              y1={y}
              x2="1000"
              y2={y}
              stroke={y === 240 ? 'rgba(100, 150, 255, 0.2)' : 'rgba(100, 150, 255, 0.05)'}
              strokeWidth="1"
            />
          ))}

          {/* Gradient fill under line */}
          <defs>
            <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={accentColor} stopOpacity="0.4" />
              <stop offset="50%" stopColor={accentColor} stopOpacity="0.2" />
              <stop offset="100%" stopColor={accentColor} stopOpacity="0.05" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {/* Area under line */}
          <path
            d={`M 0 240 ${normalized.map((v, i) => {
              const x = (i / (normalized.length - 1)) * 1000;
              const y = 240 - (v * 200);
              return `L ${x} ${y}`;
            }).join(' ')} L 1000 240 Z`}
            fill="url(#areaGradient)"
          />

          {/* Main line */}
          <path
            d={`M ${normalized.map((v, i) => {
              const x = (i / (normalized.length - 1)) * 1000;
              const y = 240 - (v * 200);
              return `${x},${y}`;
            }).join(' L ')}`}
            fill="none"
            stroke={accentColor}
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter="url(#glow)"
          />

          {/* Data points */}
          {normalized.map((value, i) => {
            const x = (i / (normalized.length - 1)) * 1000;
            const y = 240 - (value * 200);
            const isHigh = value >= threshold;
            
            return (
              <circle
                key={i}
                cx={x}
                cy={y}
                r={isHigh ? "5" : "3"}
                fill={isHigh ? accentColor : 'rgba(255, 255, 255, 0.6)'}
                stroke={isHigh ? '#fff' : 'none'}
                strokeWidth={isHigh ? "2" : "0"}
                opacity={isHigh ? "1" : "0.5"}
                style={{ cursor: 'pointer' }}
              >
                <title>{`Time: ${timestamps[i].toFixed(2)}s | Confidence: ${(scores[i] * 100).toFixed(1)}%`}</title>
              </circle>
            );
          })}
        </svg>

        {/* Time Axis */}
        <div style={{
          paddingTop: '16px',
          borderTop: '1px solid rgba(100, 150, 255, 0.2)',
          color: '#888',
          fontSize: '10px',
          fontWeight: '700',
          fontFamily: 'monospace',
          letterSpacing: '0.5px',
          position: 'relative',
          height: '24px'
        }}>
          {/* Generate time markers */}
          {Array.from({ length: 11 }, (_, i) => {
            const time = (duration * i / 10).toFixed(1);
            const position = `${i * 10}%`;
            return (
              <span key={i} style={{ 
                position: 'absolute',
                left: position,
                transform: 'translateX(-50%)',
                color: i === 0 || i === 10 ? 'rgba(100, 150, 255, 0.9)' : '#666',
                fontSize: i === 5 ? '9px' : '10px',
                fontWeight: i === 5 ? '600' : '700'
              }}>
                {i === 5 ? 'TIME' : `${time}s`}
              </span>
            );
          })}
        </div>
      </div>

      {/* Stats */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(3, 1fr)', 
        gap: '20px',
        marginTop: '28px'
      }}>
        <div style={{
          background: 'linear-gradient(135deg, rgba(100, 150, 255, 0.05) 0%, rgba(100, 150, 255, 0.02) 100%)',
          border: '1px solid rgba(100, 150, 255, 0.15)',
          borderRadius: '16px',
          padding: '24px',
          textAlign: 'center',
          transition: 'all 0.3s ease'
        }}>
          <div style={{ color: '#888', fontSize: '10px', marginBottom: '10px', fontWeight: '700', letterSpacing: '1px' }}>
            HIGH CONFIDENCE
          </div>
          <div style={{ color: accentColor, fontSize: '32px', fontWeight: '800', lineHeight: '1' }}>
            {peaks}
          </div>
          <div style={{ color: '#666', fontSize: '11px', marginTop: '8px', fontWeight: '500' }}>
            Segments
          </div>
        </div>

        <div style={{
          background: 'linear-gradient(135deg, rgba(100, 150, 255, 0.05) 0%, rgba(100, 150, 255, 0.02) 100%)',
          border: '1px solid rgba(100, 150, 255, 0.15)',
          borderRadius: '16px',
          padding: '24px',
          textAlign: 'center',
          transition: 'all 0.3s ease'
        }}>
          <div style={{ color: '#888', fontSize: '10px', marginBottom: '10px', fontWeight: '700', letterSpacing: '1px' }}>
            MAX CONFIDENCE
          </div>
          <div style={{ color: '#fff', fontSize: '32px', fontWeight: '800', lineHeight: '1' }}>
            {(maxScore * 100).toFixed(0)}%
          </div>
          <div style={{ color: '#666', fontSize: '11px', marginTop: '8px', fontWeight: '500' }}>
            Peak score
          </div>
        </div>

        <div style={{
          background: 'linear-gradient(135deg, rgba(100, 150, 255, 0.05) 0%, rgba(100, 150, 255, 0.02) 100%)',
          border: '1px solid rgba(100, 150, 255, 0.15)',
          borderRadius: '16px',
          padding: '24px',
          textAlign: 'center',
          transition: 'all 0.3s ease'
        }}>
          <div style={{ color: '#888', fontSize: '10px', marginBottom: '10px', fontWeight: '700', letterSpacing: '1px' }}>
            SEGMENTS
          </div>
          <div style={{ color: '#fff', fontSize: '32px', fontWeight: '800', lineHeight: '1' }}>
            {scores.length}
          </div>
          <div style={{ color: '#666', fontSize: '11px', marginTop: '8px', fontWeight: '500' }}>
            Analyzed
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn {
          to { opacity: 1; }
        }
      `}</style>
    </div>
  );
};

export default IntegratedGradientsMinimal;
