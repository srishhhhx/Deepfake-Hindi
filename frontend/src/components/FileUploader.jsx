import { useState } from "react";

export default function FileUploader({ onUploadSuccess, onAnalyze }) {
  const [file, setFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      if (onUploadSuccess) onUploadSuccess(selectedFile);
    }
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragOver(false);
    const droppedFile = event.dataTransfer.files[0];
    if (droppedFile) {
      setFile(droppedFile);
      if (onUploadSuccess) onUploadSuccess(droppedFile);
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const formatSize = (bytes) => {
    const mb = bytes / (1024 * 1024);
    return mb < 1 ? Math.round(bytes / 1024) + " KB" : mb.toFixed(1) + " MB";
  };

  return (
    <div style={{
      background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
      border: '1px solid #2a2a2a',
      borderRadius: '32px',
      padding: '48px',
      fontFamily: "'Montserrat', sans-serif"
    }}>
      {/* Title Section */}
      <div style={{ textAlign: 'center', marginBottom: '40px' }}>
        <div style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '12px',
          background: 'rgba(255, 255, 255, 0.05)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          borderRadius: '50px',
          padding: '12px 28px',
          marginBottom: '16px'
        }}>
          <svg width="20" height="20" fill="#fff" viewBox="0 0 20 20">
            <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" />
          </svg>
          <span style={{ color: '#fff', fontSize: '14px', fontWeight: '700', letterSpacing: '1px' }}>
            AUDIO ANALYSIS
          </span>
        </div>
        <h2 style={{ color: '#fff', fontSize: '32px', fontWeight: '700', marginTop: '16px' }}>Upload Audio File</h2>
        <p style={{ color: '#666', fontSize: '14px', marginTop: '8px' }}>WAV, MP3, or FLAC • Max 10MB</p>
      </div>

      {/* File Preview or Upload Area */}
      {file ? (
        <div>
          <div style={{
            background: 'rgba(255, 255, 255, 0.02)',
            border: '1px solid #2a2a2a',
            borderRadius: '16px',
            padding: '24px',
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
            marginBottom: '32px'
          }}>
            <div style={{
              width: '56px',
              height: '56px',
              background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
              border: '1px solid #333',
              borderRadius: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0
            }}>
              <svg width="28" height="28" fill="white" viewBox="0 0 20 20">
                <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" />
              </svg>
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ color: '#fff', fontSize: '16px', fontWeight: '600', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{file.name}</div>
              <div style={{ color: '#666', fontSize: '14px', marginTop: '4px' }}>{formatSize(file.size)}</div>
            </div>
            <button
              onClick={() => setFile(null)}
              style={{
                width: '40px',
                height: '40px',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '10px',
                color: '#ef4444',
                fontSize: '24px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0,
                transition: 'all 0.2s'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(239, 68, 68, 0.2)';
                e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.5)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)';
                e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.3)';
              }}
            >
              ×
            </button>
          </div>

          {onAnalyze && (
            <button
              onClick={onAnalyze}
              className="shimmer-btn"
              style={{
                width: '100%',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
                border: '1px solid #333',
                borderRadius: '16px',
                padding: '20px',
                color: 'white',
                fontSize: '18px',
                fontWeight: '700',
                cursor: 'pointer',
                position: 'relative',
                overflow: 'hidden',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)'
              }}
            >
              <div className="shimmer" />
              <span style={{ position: 'relative', zIndex: 1 }}>Analyze Audio</span>
            </button>
          )}
        </div>
      ) : (
        <div>
          <div
            style={{
              border: isDragOver ? '2px dashed #fff' : '2px dashed #333',
              background: isDragOver ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.02)',
              borderRadius: '24px',
              padding: '80px 48px',
              textAlign: 'center',
              transition: 'all 0.3s',
              cursor: 'pointer'
            }}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <input
              type="file"
              onChange={handleFileChange}
              id="file-input"
              style={{ display: 'none' }}
              accept="audio/*,.wav,.mp3,.flac"
            />
            <label htmlFor="file-input" style={{ cursor: 'pointer', display: 'block' }}>
              <div style={{
                width: '80px',
                height: '80px',
                background: 'linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%)',
                border: '1px solid #333',
                borderRadius: '20px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 24px',
                boxShadow: '0 10px 40px rgba(0, 0, 0, 0.5)'
              }}>
                <svg width="40" height="40" stroke="white" fill="none" viewBox="0 0 24 24" strokeWidth="2">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2 -2v-2" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M7 9l5 -5l5 5" />
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 4l0 12" />
                </svg>
              </div>

              <div style={{ color: '#aaa', fontSize: '16px' }}>
                Drop files here or <span style={{ color: '#fff', fontWeight: '600' }}>browse</span>
              </div>
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
