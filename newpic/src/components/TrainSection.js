import React, { useCallback, useMemo, useState } from 'react';
import Button from 'react-bootstrap/Button';
import Modal from 'react-bootstrap/Modal';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { useImageMap } from '../contexts/ImageMapContext';
import { useSharedClassifier } from '../contexts/ImageClassifierContext';
import AdvancedSettings from './AdvancedSettings';
import { DEFAULT_CONFIG } from '../ml/config/defaults';

const TrainSection = () => {
  const { imageMap, setIsTrained, setActiveWebcamClass, serverUrl, setServerUrl } = useImageMap();
  const { train, trainOnServer, loading, progress, message, error, validationMetrics, trainingSummary } = useSharedClassifier();

  const [showSettings, setShowSettings] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [trainingMode, setTrainingMode] = useState('browser');
  const [trainingOptions, setTrainingOptions] = useState({
    transferModel: DEFAULT_CONFIG.transferModel,
    learningRate: 0.0001,
    optimizer: 'adam',
    epochs: 20,
    batchSizeFraction: 0.4,
    useValidation: false
  });

  const canTrain = useMemo(() => {
    const labels = Object.keys(imageMap);
    if (labels.length < 2) return false;
    for (const label of labels) {
      if (imageMap[label].length < 2) return false;
    }
    return true;
  }, [imageMap]);

  const handleTrain = useCallback(async () => {
    if (!canTrain) return;
    // Hide webcam from all classes when training starts
    setActiveWebcamClass(null);
    setIsTrained(false);
    const success = trainingMode === 'server'
      ? await trainOnServer(imageMap, trainingOptions, serverUrl)
      : await train(imageMap, trainingOptions);
    if (success) {
      setIsTrained(true);
    }
  }, [canTrain, train, trainOnServer, imageMap, trainingOptions, trainingMode, serverUrl, setIsTrained, setActiveWebcamClass]);

  return (
    <section style={{ marginBottom: '60px' }}>
      <h2 style={{
        fontSize: '20px',
        fontWeight: '500',
        fontFamily: 'Poppins-Regular',
        marginBottom: '20px',
        color: '#1a1a1d'
      }}>
        2. Train Model
      </h2>

      <div style={{
        backgroundColor: 'white',
        borderRadius: '12px',
        padding: '32px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        position: 'relative'
      }}>
        {/* Settings Cog Icon */}
        <button
          onClick={() => setShowSettings(true)}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: '50%',
            transition: 'background-color 0.2s'
          }}
          onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f0f0f0'}
          onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
          title="Training Settings"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M12 1v6m0 6v6m-9-9h6m6 0h6m-3.8-6.8l-1.4 1.4m-8.4 8.4l-1.4 1.4m11.3 0l-1.4-1.4m-8.4-8.4l-1.4-1.4"></path>
          </svg>
        </button>
        {/* Training Mode Toggle */}
        <div style={{ marginBottom: '20px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '10px' }}>
          <div style={{ display: 'flex', gap: '0', border: '1px solid #ddd', borderRadius: '6px', overflow: 'hidden' }}>
            {['browser', 'server'].map((mode) => (
              <button
                key={mode}
                onClick={() => setTrainingMode(mode)}
                style={{
                  padding: '6px 20px',
                  border: 'none',
                  cursor: 'pointer',
                  fontFamily: 'Poppins-Regular',
                  fontSize: '13px',
                  fontWeight: trainingMode === mode ? '600' : '400',
                  backgroundColor: trainingMode === mode ? '#4285f4' : 'white',
                  color: trainingMode === mode ? 'white' : '#666',
                  transition: 'all 0.15s'
                }}
              >
                {mode.charAt(0).toUpperCase() + mode.slice(1)}
              </button>
            ))}
          </div>

          {trainingMode === 'server' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <label style={{ fontSize: '12px', color: '#666', fontFamily: 'Poppins-Regular' }}>Server URL:</label>
              <input
                type="text"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                disabled={loading}
                style={{
                  padding: '4px 10px',
                  fontSize: '12px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  width: '220px'
                }}
              />
            </div>
          )}
        </div>

        {!canTrain && (
          <p style={{ color: '#666', marginBottom: '20px', fontSize: '14px' }}>
            Add at least 2 classes with 2+ samples each to train
          </p>
        )}

        {error && (
          <div style={{
            backgroundColor: '#fee',
            color: '#c33',
            padding: '12px',
            borderRadius: '6px',
            marginBottom: '20px',
            fontSize: '14px'
          }}>
            {error}
          </div>
        )}

        {loading && (
          <div style={{ marginBottom: '20px', width: '100%', maxWidth: '500px' }}>
            <ProgressBar
              now={progress}
              style={{ height: '8px', marginBottom: '10px' }}
            />
            <p style={{ fontSize: '14px', color: '#666', margin: 0 }}>
              {message}
            </p>
            {validationMetrics && (
              <div style={{
                marginTop: '16px',
                padding: '12px',
                backgroundColor: '#f8f9fa',
                borderRadius: '6px',
                fontSize: '13px',
                fontFamily: 'Poppins-Regular'
              }}>
                <div style={{ marginBottom: '4px' }}>
                  <strong>Validation Metrics:</strong>
                </div>
                <div style={{ color: '#666' }}>
                  Accuracy: <span style={{ color: '#34a853', fontWeight: '500' }}>
                    {(validationMetrics.val_acc * 100).toFixed(1)}%
                  </span>
                </div>
                <div style={{ color: '#666' }}>
                  Loss: {validationMetrics.val_loss.toFixed(5)}
                </div>
              </div>
            )}
          </div>
        )}

        <Button
          onClick={handleTrain}
          disabled={!canTrain || loading}
          style={{
            backgroundColor: '#4285f4',
            border: 'none',
            padding: '12px 48px',
            fontSize: '16px',
            borderRadius: '24px',
            fontFamily: 'Poppins-Regular',
            fontWeight: '500'
          }}
        >
          {loading ? 'Training...' : 'Train Model'}
        </Button>

        {!loading && message && !error && (
          <div style={{ marginTop: '16px', width: '100%', maxWidth: '500px' }}>
            <p style={{ color: '#34a853', fontSize: '14px', margin: '0 0 12px 0' }}>
              ✓ {message}
            </p>

            {trainingSummary && (
              <div style={{
                backgroundColor: '#f8f9fa',
                borderRadius: '8px',
                overflow: 'hidden',
                border: '1px solid #e0e0e0'
              }}>
                <button
                  onClick={() => setShowSummary(!showSummary)}
                  style={{
                    width: '100%',
                    padding: '12px 16px',
                    backgroundColor: 'transparent',
                    border: 'none',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    fontFamily: 'Poppins-Regular',
                    fontSize: '13px',
                    fontWeight: '500',
                    color: '#1a1a1d'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#e9ecef'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                >
                  <span>Training Summary</span>
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                    style={{
                      transform: showSummary ? 'rotate(180deg)' : 'rotate(0deg)',
                      transition: 'transform 0.2s'
                    }}
                  >
                    <path
                      d="M4 6L8 10L12 6"
                      stroke="#666"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>

                {showSummary && (
                  <div style={{
                    padding: '16px',
                    borderTop: '1px solid #e0e0e0',
                    fontFamily: 'Poppins-Regular',
                    fontSize: '13px'
                  }}>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                      <div>
                        <div style={{ color: '#666', marginBottom: '4px' }}>Duration</div>
                        <div style={{ fontWeight: '500' }}>{trainingSummary.duration}s</div>
                      </div>
                      <div>
                        <div style={{ color: '#666', marginBottom: '4px' }}>Epochs</div>
                        <div style={{ fontWeight: '500' }}>{trainingSummary.epochs}</div>
                      </div>
                      <div>
                        <div style={{ color: '#666', marginBottom: '4px' }}>Classes</div>
                        <div style={{ fontWeight: '500' }}>{trainingSummary.numClasses}</div>
                      </div>
                      <div>
                        <div style={{ color: '#666', marginBottom: '4px' }}>Total Samples</div>
                        <div style={{ fontWeight: '500' }}>{trainingSummary.totalSamples}</div>
                      </div>
                    </div>

                    {trainingSummary.useValidation && (
                      <div style={{
                        marginTop: '12px',
                        paddingTop: '12px',
                        borderTop: '1px solid #e0e0e0'
                      }}>
                        <div style={{ color: '#666', marginBottom: '8px', fontWeight: '500' }}>Data Split</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                          <div>
                            <div style={{ color: '#666', fontSize: '12px' }}>Training</div>
                            <div style={{ fontWeight: '500' }}>{trainingSummary.trainSize} samples</div>
                          </div>
                          <div>
                            <div style={{ color: '#666', fontSize: '12px' }}>Validation</div>
                            <div style={{ fontWeight: '500' }}>{trainingSummary.valSize} samples</div>
                          </div>
                        </div>
                      </div>
                    )}

                    {trainingSummary.finalMetrics && trainingSummary.finalMetrics.loss && (
                      <div style={{
                        marginTop: '12px',
                        paddingTop: '12px',
                        borderTop: '1px solid #e0e0e0'
                      }}>
                        <div style={{ color: '#666', marginBottom: '8px', fontWeight: '500' }}>Final Metrics</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                          <div>
                            <div style={{ color: '#666', fontSize: '12px' }}>Training Loss</div>
                            <div style={{ fontWeight: '500' }}>{trainingSummary.finalMetrics.loss.toFixed(5)}</div>
                          </div>
                          <div>
                            <div style={{ color: '#666', fontSize: '12px' }}>Training Accuracy</div>
                            <div style={{ fontWeight: '500', color: '#34a853' }}>
                              {(trainingSummary.finalMetrics.acc * 100).toFixed(1)}%
                            </div>
                          </div>
                          {trainingSummary.useValidation && (
                            <>
                              <div>
                                <div style={{ color: '#666', fontSize: '12px' }}>Validation Loss</div>
                                <div style={{ fontWeight: '500' }}>
                                  {trainingSummary.finalMetrics.val_loss !== undefined
                                    ? trainingSummary.finalMetrics.val_loss.toFixed(5)
                                    : 'N/A'}
                                </div>
                              </div>
                              <div>
                                <div style={{ color: '#666', fontSize: '12px' }}>Validation Accuracy</div>
                                <div style={{ fontWeight: '500', color: '#34a853' }}>
                                  {trainingSummary.finalMetrics.val_acc !== undefined
                                    ? `${(trainingSummary.finalMetrics.val_acc * 100).toFixed(1)}%`
                                    : 'N/A'}
                                </div>
                              </div>
                            </>
                          )}
                        </div>
                      </div>
                    )}

                    <div style={{
                      marginTop: '12px',
                      paddingTop: '12px',
                      borderTop: '1px solid #e0e0e0'
                    }}>
                      <div style={{ color: '#666', marginBottom: '8px', fontWeight: '500' }}>Hyperparameters</div>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                        <div>
                          <div style={{ color: '#666', fontSize: '12px' }}>Optimizer</div>
                          <div style={{ fontWeight: '500', textTransform: 'uppercase' }}>{trainingSummary.optimizer}</div>
                        </div>
                        <div>
                          <div style={{ color: '#666', fontSize: '12px' }}>Learning Rate</div>
                          <div style={{ fontWeight: '500' }}>{trainingSummary.learningRate}</div>
                        </div>
                        <div>
                          <div style={{ color: '#666', fontSize: '12px' }}>Batch Size Fraction</div>
                          <div style={{ fontWeight: '500' }}>{(trainingSummary.batchSizeFraction * 100).toFixed(0)}%</div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Settings Modal */}
      <Modal show={showSettings} onHide={() => setShowSettings(false)} centered>
        <Modal.Header closeButton style={{ fontFamily: 'Poppins-Regular' }}>
          <Modal.Title style={{ fontSize: '18px', fontWeight: '500' }}>
            Training Settings
          </Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <AdvancedSettings
            options={trainingOptions}
            onChange={setTrainingOptions}
            disabled={loading}
          />
        </Modal.Body>
        <Modal.Footer>
          <Button
            variant="primary"
            onClick={() => setShowSettings(false)}
            style={{
              fontFamily: 'Poppins-Regular',
              backgroundColor: '#4285f4',
              border: 'none'
            }}
          >
            Done
          </Button>
        </Modal.Footer>
      </Modal>
    </section>
  );
};

export default TrainSection;
