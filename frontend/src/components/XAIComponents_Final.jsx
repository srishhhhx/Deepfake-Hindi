import React from 'react';

// Import clean, minimal components
import {
  IntegratedGradientsCard,
  MelSpectrogramCard,
  SHAPCard,
} from './XAI_Clean';
import { IntegratedGradientsMinimal } from './IntegratedGradientsMinimal';

// Main XAI Visualizations Component with staggered delays
const XAIVisualizations = ({ xaiData, isFake, fileName }) => {
  if (!xaiData) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#888' }}>
        <p>No XAI data available</p>
      </div>
    );
  }

  const accentColor = isFake ? '#ef4444' : '#22c55e';
  const basicXAI = xaiData.basic_xai || {};
  const advancedXAI = xaiData.advanced_xai || {};

  // Check if we have any data to display
  const hasData = (basicXAI && Object.keys(basicXAI).length > 0) || 
                  (advancedXAI && Object.keys(advancedXAI).length > 0);
  
  if (!hasData) {
    return (
      <div style={{ 
        padding: '60px', 
        textAlign: 'center', 
        color: '#888',
        background: 'rgba(255, 255, 255, 0.02)',
        borderRadius: '24px',
        marginTop: '40px'
      }}>
        <p style={{ fontSize: '18px', marginBottom: '12px' }}>⚠️ XAI Analysis Incomplete</p>
        <p style={{ fontSize: '14px', color: '#666' }}>The explainability analysis did not return valid data. Please try again.</p>
      </div>
    );
  }

  // Staggered delays for smooth sequential appearance
  const baseDelay = 0;
  const delayIncrement = 1400; // 1.4 seconds between each graph

  return (
    <div style={{ marginTop: '80px' }}>
      {/* Section Header */}
      <div style={{
        textAlign: 'center',
        marginBottom: '32px',
        padding: '24px 32px',
        background: 'rgba(255, 255, 255, 0.02)',
        borderRadius: '16px',
        border: '1px solid rgba(255, 255, 255, 0.08)'
      }}>
        <h2 style={{
          color: '#fff',
          fontSize: '28px',
          fontWeight: '700',
          marginBottom: '8px',
          letterSpacing: '-0.5px'
        }}>
          Explainable AI Analysis
        </h2>
        <p style={{ color: '#888', fontSize: '14px', maxWidth: '700px', margin: '0 auto', lineHeight: '1.6' }}>
          Multiple analysis methods reveal different aspects of the model's reasoning process.
        </p>
        <div style={{
          marginTop: '12px',
          color: '#666',
          fontSize: '12px',
          fontFamily: 'monospace',
          fontWeight: '600'
        }}>
          Processing Time: {xaiData.processing_time_ms || 0}ms
        </div>
      </div>

      {/* 1. TEMPORAL ATTRIBUTION - Time Segment Analysis */}
      {basicXAI.temporal_heatmap && (
        <IntegratedGradientsMinimal
          data={advancedXAI.integrated_gradients}
          temporalData={basicXAI.temporal_heatmap}
          accentColor={accentColor}
          delay={baseDelay + delayIncrement * 0}
          audioDuration={basicXAI.temporal_heatmap?.timestamps?.[basicXAI.temporal_heatmap.timestamps.length - 1]}
          isFake={isFake}
        />
      )}

      {/* 2. INTEGRATED GRADIENTS - Feature Attribution */}
      {advancedXAI.integrated_gradients && (
        <IntegratedGradientsCard
          data={advancedXAI.integrated_gradients}
          accentColor={accentColor}
          delay={baseDelay + delayIncrement * 1}
          isFake={isFake}
        />
      )}

      {/* 3. MEL SPECTROGRAM */}
      {basicXAI.frequency_contribution && (
        <MelSpectrogramCard
          data={basicXAI.frequency_contribution}
          accentColor={accentColor}
          delay={baseDelay + delayIncrement * 2}
          isFake={isFake}
        />
      )}

      {/* 4. EXPERT SHAP */}
      {advancedXAI.shap_approximation && (
        <SHAPCard
          data={advancedXAI.shap_approximation}
          accentColor={accentColor}
          delay={baseDelay + delayIncrement * 3}
          isFake={isFake}
        />
      )}


      {/* Global CSS Animations */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.5; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.05); }
        }
      `}</style>
    </div>
  );
};

export default XAIVisualizations;
