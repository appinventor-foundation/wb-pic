import React, { useState, useCallback, useRef } from 'react';
import Button from 'react-bootstrap/Button';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { useImageMap } from '../contexts/ImageMapContext';
import { useZipImport } from '../hooks/useZipImport';
import ClassSection from './ClassSection';
import TrainSection from './TrainSection';
import PreviewSection from './PreviewSection';
import ExportSection from './ExportSection';

const TeachableMachineLayout = () => {
  const { imageMap, addLabel, updateImageMap, serverUrl } = useImageMap();
  const [newClassName, setNewClassName] = useState('');
  const [showNameInput, setShowNameInput] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const zipInputRef = useRef(null);
  const elapsedIntervalRef = useRef(null);

  React.useEffect(() => () => clearInterval(elapsedIntervalRef.current), []);

  const { handleZipImport } = useZipImport(
    updateImageMap,
    () => {
      setIsImporting(true);
      setElapsedSeconds(0);
      elapsedIntervalRef.current = setInterval(() => setElapsedSeconds(s => s + 1), 1000);
    },
    () => {
      setIsImporting(false);
      setImportProgress(null);
      clearInterval(elapsedIntervalRef.current);
    },
    serverUrl,
    (progress) => setImportProgress(progress)
  );

  const handleAddClass = useCallback(() => {
    setShowNameInput(true);
  }, []);

  const handleCreateClass = useCallback(() => {
    if (newClassName.trim()) {
      addLabel(newClassName.trim());
      setNewClassName('');
      setShowNameInput(false);
    }
  }, [newClassName, addLabel]);

  const handleKeyPress = useCallback((e) => {
    if (e.key === 'Enter') {
      handleCreateClass();
    } else if (e.key === 'Escape') {
      setShowNameInput(false);
      setNewClassName('');
    }
  }, [handleCreateClass]);

  return (
    <div style={{
      maxWidth: '1400px',
      margin: '0 auto',
      padding: '40px 20px',
      backgroundColor: '#f8f9fa'
    }}>
      {isImporting && (
        <div style={{
          position: 'fixed', inset: 0,
          backgroundColor: 'rgba(0,0,0,0.55)',
          zIndex: 9999,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: 'Poppins-Regular'
        }}>
          <div style={{
            backgroundColor: 'white', borderRadius: '16px',
            padding: '40px 48px', width: '420px', maxWidth: '90vw',
            boxShadow: '0 8px 32px rgba(0,0,0,0.25)',
            display: 'flex', flexDirection: 'column', gap: '16px'
          }}>
            <div style={{ fontSize: '13px', color: '#888', textAlign: 'center', wordBreak: 'break-all' }}>
              {importProgress?.filename ?? 'Reading file...'}
            </div>
            <div style={{ fontSize: '18px', fontWeight: '500', color: '#1a1a1d', textAlign: 'center' }}>
              {importProgress?.phase === 'upload' && 'Uploading...'}
              {importProgress?.phase === 'extract' && 'Extracting on server...'}
              {!importProgress && 'Starting...'}
            </div>
            <ProgressBar
              now={
                importProgress?.phase === 'upload' ? importProgress.uploadPct :
                importProgress?.phase === 'extract' && importProgress.total > 0
                  ? Math.round((importProgress.done / importProgress.total) * 100) : 0
              }
              style={{ height: '8px' }}
            />
            <div style={{ fontSize: '13px', color: '#666', textAlign: 'center' }}>
              {importProgress?.phase === 'upload' && `${importProgress.uploadPct}% of ${(importProgress.fileSize / 1024 / 1024).toFixed(1)} MB`}
              {importProgress?.phase === 'extract' && importProgress.total > 0 && `${importProgress.done} / ${importProgress.total} files`}
              {(!importProgress || (importProgress?.phase === 'extract' && importProgress.total === 0)) && '\u00a0'}
            </div>
            <div style={{ fontSize: '12px', color: '#aaa', textAlign: 'center' }}>
              {elapsedSeconds}s elapsed
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div style={{ marginBottom: '40px', textAlign: 'center' }}>
        <h1 style={{
          fontSize: '32px',
          fontWeight: '400',
          fontFamily: 'Poppins-Regular',
          color: '#1a1a1d',
          marginBottom: '10px'
        }}>
          Personal Image Classifier
        </h1>
        <p style={{ fontSize: '16px', color: '#666', marginBottom: '16px' }}>
          Train a computer to recognize your own images
        </p>
        <button
          onClick={() => {
            console.log('[Import] button clicked, ref:', zipInputRef.current);
            zipInputRef.current?.click();
            console.log('[Import] .click() called');
          }}
          disabled={isImporting}
          style={{
            padding: '5px 10px',
            fontSize: '13px',
            fontFamily: 'Poppins-Regular',
            color: isImporting ? '#aaa' : '#6c757d',
            border: `1px solid ${isImporting ? '#ccc' : '#6c757d'}`,
            borderRadius: '4px',
            cursor: isImporting ? 'not-allowed' : 'pointer',
            background: 'none'
          }}
        >
          {isImporting ? 'Importing...' : 'Import Dataset (ZIP)'}
        </button>
        <input
          ref={zipInputRef}
          type="file"
          style={{ position: 'fixed', top: '-9999px', left: '-9999px' }}
          onChange={(e) => {
            console.log('[Import] onChange fired, files:', e.target.files);
            handleZipImport(e);
          }}
        />
      </div>

      {/* Training Section */}
      <section style={{ marginBottom: '60px' }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: '500',
          fontFamily: 'Poppins-Regular',
          marginBottom: '20px',
          color: '#1a1a1d'
        }}>
          1. Gather Samples
        </h2>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {Object.keys(imageMap).map((className) => (
            <ClassSection key={className} className={className} />
          ))}

          {/* Add Class Section */}
          {showNameInput ? (
            <div style={{
              backgroundColor: 'white',
              borderRadius: '12px',
              padding: '30px',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              border: '2px solid #4285f4'
            }}>
              <input
                type="text"
                value={newClassName}
                onChange={(e) => setNewClassName(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Enter class name (e.g., Cat, Dog, Happy, Sad...)"
                autoFocus
                style={{
                  width: '100%',
                  padding: '12px',
                  fontSize: '16px',
                  border: '1px solid #ddd',
                  borderRadius: '6px',
                  fontFamily: 'Poppins-Regular'
                }}
              />
              <div style={{ marginTop: '10px', display: 'flex', gap: '10px' }}>
                <Button variant="primary" onClick={handleCreateClass}>
                  Create Class
                </Button>
                <Button variant="outline-secondary" onClick={() => {
                  setShowNameInput(false);
                  setNewClassName('');
                }}>
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <button
              onClick={handleAddClass}
              style={{
                backgroundColor: 'white',
                border: '2px dashed #ddd',
                borderRadius: '12px',
                padding: '40px',
                cursor: 'pointer',
                transition: 'all 0.2s',
                fontSize: '16px',
                color: '#666',
                fontFamily: 'Poppins-Regular'
              }}
              onMouseEnter={(e) => {
                e.target.style.borderColor = '#4285f4';
                e.target.style.backgroundColor = '#f8f9fa';
              }}
              onMouseLeave={(e) => {
                e.target.style.borderColor = '#ddd';
                e.target.style.backgroundColor = 'white';
              }}
            >
              + Add a class
            </button>
          )}
        </div>
      </section>

      {/* Train Section */}
      <TrainSection />

      {/* Preview Section */}
      <PreviewSection />

      {/* Export Section */}
      <ExportSection />
    </div>
  );
};

export default TeachableMachineLayout;
