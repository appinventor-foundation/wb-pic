import React, { useRef, useState, useCallback } from 'react';
import Button from 'react-bootstrap/Button';
import Alert from 'react-bootstrap/Alert';
import * as JSZip from 'jszip';
import * as tf from '@tensorflow/tfjs';
import { useImageMap } from '../contexts/ImageMapContext';
import { useSharedClassifier } from '../contexts/ImageClassifierContext';

const ExportSection = () => {
  const { imageMap, isTrained, setLoadedModel, updateImageMap } = useImageMap();
  const { exportModel, exportData } = useSharedClassifier();
  const [uploadError, setUploadError] = useState(null);
  const [uploadSuccess, setUploadSuccess] = useState(null);

  const modelInputRef = useRef(null);
  const dataInputRef = useRef(null);

  const handleExportModel = useCallback(async () => {
    if (!isTrained) {
      setUploadError('No trained model to export. Train a model first.');
      return;
    }
    try {
      await exportModel();
      setUploadSuccess('Model exported successfully!');
      setTimeout(() => setUploadSuccess(null), 3000);
    } catch (error) {
      setUploadError(`Export failed: ${error.message}`);
    }
  }, [isTrained, exportModel]);

  const handleExportData = useCallback(async () => {
    if (Object.keys(imageMap).length === 0) {
      setUploadError('No training data to export. Add some classes first.');
      return;
    }
    try {
      await exportData(imageMap);
      setUploadSuccess('Training data exported successfully!');
      setTimeout(() => setUploadSuccess(null), 3000);
    } catch (error) {
      setUploadError(`Export failed: ${error.message}`);
    }
  }, [imageMap, exportData]);

  const uploadModel = useCallback(() => {
    modelInputRef.current.click();
  }, []);

  const uploadData = useCallback(() => {
    dataInputRef.current.click();
  }, []);

  const handleModel = useCallback(async () => {
    try {
      setUploadError(null);
      setUploadSuccess(null);
      const zip = new JSZip();
      let data = await zip.loadAsync(modelInputRef.current.files[0]);
      const weightData = new File([await data.files['model.weights.bin'].async('blob')], "model.weights.bin");
      const topologyWeightsJSON = new File([await data.files['model.json'].async('blob')], "model.json");
      const model = await tf.loadLayersModel(tf.io.browserFiles([topologyWeightsJSON, weightData]));
      model.summary();
      setLoadedModel(model);
      setUploadSuccess('Model uploaded successfully!');
      setTimeout(() => setUploadSuccess(null), 3000);
    } catch (error) {
      setUploadError(`Failed to load model: ${error.message}`);
    }
  }, [setLoadedModel]);

  const handleData = useCallback(async () => {
    try {
      setUploadError(null);
      setUploadSuccess(null);
      const zip = new JSZip();
      let data = await zip.loadAsync(dataInputRef.current.files[0]);
      const loadedMap = JSON.parse(await data.files['images.json'].async('string'));
      updateImageMap(loadedMap);
      setUploadSuccess('Training data uploaded successfully!');
      setTimeout(() => setUploadSuccess(null), 3000);
    } catch (error) {
      setUploadError(`Failed to load data: ${error.message}`);
    }
  }, [updateImageMap]);

  const hasData = Object.keys(imageMap).length > 0;

  return (
    <section style={{ marginBottom: '60px' }}>
      <h2 style={{
        fontSize: '20px',
        fontWeight: '500',
        fontFamily: 'Poppins-Regular',
        marginBottom: '20px',
        color: '#1a1a1d'
      }}>
        4. Export
      </h2>

      {uploadError && (
        <Alert variant="danger" dismissible onClose={() => setUploadError(null)} style={{ marginBottom: '20px' }}>
          {uploadError}
        </Alert>
      )}

      {uploadSuccess && (
        <Alert variant="success" dismissible onClose={() => setUploadSuccess(null)} style={{ marginBottom: '20px' }}>
          {uploadSuccess}
        </Alert>
      )}

      <div style={{
        backgroundColor: 'white',
        borderRadius: '12px',
        padding: '32px',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '32px'
      }}>
        {/* Export Section */}
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: '500', marginBottom: '16px', fontFamily: 'Poppins-Regular' }}>
            Download
          </h3>
          <p style={{ fontSize: '14px', color: '#666', marginBottom: '20px' }}>
            Save your trained model and training data
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <Button
              variant="primary"
              onClick={handleExportModel}
              disabled={!isTrained}
              style={{ width: '100%' }}
            >
              Export Model
            </Button>
            <Button
              variant="outline-primary"
              onClick={handleExportData}
              disabled={!hasData}
              style={{ width: '100%' }}
            >
              Export Training Data
            </Button>
          </div>
        </div>

        {/* Upload Section */}
        <div>
          <h3 style={{ fontSize: '16px', fontWeight: '500', marginBottom: '16px', fontFamily: 'Poppins-Regular' }}>
            Upload
          </h3>
          <p style={{ fontSize: '14px', color: '#666', marginBottom: '20px' }}>
            Load previously saved files
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <Button
              variant="outline-secondary"
              onClick={uploadModel}
              style={{ width: '100%' }}
            >
              Upload Model
            </Button>
            <input
              type="file"
              ref={modelInputRef}
              onChange={handleModel}
              accept=".zip,.mdl"
              style={{ display: 'none' }}
            />
            <Button
              variant="outline-secondary"
              onClick={uploadData}
              style={{ width: '100%' }}
            >
              Upload Training Data
            </Button>
            <input
              type="file"
              ref={dataInputRef}
              onChange={handleData}
              accept=".zip"
              style={{ display: 'none' }}
            />
          </div>
        </div>
      </div>
    </section>
  );
};

export default ExportSection;
