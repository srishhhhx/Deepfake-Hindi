import { useState } from "react";
import FileUploader from "./components/FileUploader";
import ResultDisplay from "./components/ResultDisplay";
import GlobalXAI from "./pages/GlobalXAI";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");
  const [showGlobalXAI, setShowGlobalXAI] = useState(false);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setErr("");
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setErr("");
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/infer`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const json = await r.json();
      
      console.log('Backend response:', json); // Debug log
      
      // Check for errors
      if (json.error) {
        setErr(json.error + (json.hint ? ' ' + json.hint : ''));
        return;
      }
      
      // Check if language gate rejected the audio
      if (json.error_code === "not_hindi") {
        setResult(json); // Still set result to show rejection UI
      } else {
        setResult(json);
      }
    } catch (e) {
      console.error('Error:', e);
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setErr("");
  };

  // If showing Global XAI, render that instead
  if (showGlobalXAI) {
    return <GlobalXAI onClose={() => setShowGlobalXAI(false)} />;
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%)',
      padding: '60px 40px',
      fontFamily: "'Montserrat', sans-serif"
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: '80px' }}>
          <h1 style={{
            color: '#fff',
            fontSize: '48px',
            fontWeight: '800',
            letterSpacing: '-1px',
            marginBottom: '32px'
          }}>
            Hindi Audio Deepfake Detector
          </h1>
          
          {/* Global XAI Button */}
          <button
            onClick={() => setShowGlobalXAI(true)}
            style={{
              padding: '14px 32px',
              background: 'linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%)',
              color: '#fff',
              border: '1px solid #2a2a2a',
              borderRadius: '12px',
              fontSize: '14px',
              fontWeight: '700',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.6)',
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
            View Global XAI Analysis
          </button>
        </div>

        {/* Main Content */}
        <div className="flex justify-center">
          {loading ? (
            <div style={{ textAlign: 'center', padding: '80px 40px' }}>
              <div style={{
                width: '60px',
                height: '60px',
                border: '4px solid #333',
                borderTop: '4px solid #fff',
                borderRadius: '50%',
                margin: '0 auto 24px',
                animation: 'spin 1s linear infinite'
              }} />
              <p style={{ color: '#888', fontSize: '16px' }}>Analyzing audio...</p>
            </div>
          ) : !result ? (
            <div className="w-full max-w-md">
              <FileUploader 
                onUploadSuccess={handleFileSelect}
                onAnalyze={handleAnalyze}
              />
              
              {err && (
                <div className="mt-4 bg-gradient-to-r from-[#2a0a0a] to-[#1a0505] border border-[#ff6b6b]/20 rounded-xl p-3">
                  <div className="text-[#ff6b6b] text-sm font-semibold flex items-center gap-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <span>{err}</span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <ResultDisplay 
              result={result} 
              fileName={file?.name} 
              fileSize={file?.size}
              onReset={handleReset}
            />
          )}
        </div>

      </div>
    </div>
  );
}
