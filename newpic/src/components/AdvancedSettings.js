import React from 'react';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import { TRANSFER_MODEL_OPTIONS, DEFAULT_CONFIG } from '../ml/config/defaults';

const DEFAULT_OPTIONS = {
  transferModel: DEFAULT_CONFIG.transferModel,
  learningRate: 0.0001,
  optimizer: 'adam',
  epochs: 20,
  batchSizeFraction: 0.4,
  useValidation: false
};

const AdvancedSettings = ({ options, onChange, disabled }) => {
  const handleChange = (field, value) => {
    onChange({
      ...options,
      [field]: value
    });
  };

  const handleReset = () => {
    onChange(DEFAULT_OPTIONS);
  };

  return (
    <Form>
      <Form.Group className="mb-3">
        <Form.Label style={{ fontFamily: 'Poppins-Regular', fontSize: '14px' }}>
          Transfer Model
        </Form.Label>
        <Form.Select
          value={options.transferModel?.name || 'mobilenet_v1'}
          onChange={(e) => handleChange('transferModel', TRANSFER_MODEL_OPTIONS[e.target.value])}
          disabled={disabled}
          style={{ fontFamily: 'Poppins-Regular' }}
        >
          {Object.values(TRANSFER_MODEL_OPTIONS).map(m => (
            <option key={m.name} value={m.name}>{m.label}</option>
          ))}
        </Form.Select>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label style={{ fontFamily: 'Poppins-Regular', fontSize: '14px' }}>
          Learning Rate
        </Form.Label>
        <Form.Control
          type="number"
          step="0.00001"
          min="0.00001"
          max="0.1"
          value={options.learningRate}
          onChange={(e) => handleChange('learningRate', parseFloat(e.target.value))}
          disabled={disabled}
          style={{ fontFamily: 'Poppins-Regular' }}
        />
        <Form.Text className="text-muted" style={{ fontFamily: 'Poppins-Regular', fontSize: '12px' }}>
          Default: 0.0001
        </Form.Text>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label style={{ fontFamily: 'Poppins-Regular', fontSize: '14px' }}>
          Optimizer
        </Form.Label>
        <Form.Select
          value={options.optimizer}
          onChange={(e) => handleChange('optimizer', e.target.value)}
          disabled={disabled}
          style={{ fontFamily: 'Poppins-Regular' }}
        >
          <option value="adam">Adam</option>
          <option value="sgd">SGD</option>
          <option value="adagrad">Adagrad</option>
          <option value="adadelta">Adadelta</option>
        </Form.Select>
        <Form.Text className="text-muted" style={{ fontFamily: 'Poppins-Regular', fontSize: '12px' }}>
          Default: adam
        </Form.Text>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label style={{ fontFamily: 'Poppins-Regular', fontSize: '14px' }}>
          Epochs
        </Form.Label>
        <Form.Control
          type="number"
          min="1"
          max="200"
          value={options.epochs}
          onChange={(e) => handleChange('epochs', parseInt(e.target.value))}
          disabled={disabled}
          style={{ fontFamily: 'Poppins-Regular' }}
        />
        <Form.Text className="text-muted" style={{ fontFamily: 'Poppins-Regular', fontSize: '12px' }}>
          Default: 20
        </Form.Text>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Label style={{ fontFamily: 'Poppins-Regular', fontSize: '14px' }}>
          Batch Size Fraction
        </Form.Label>
        <Form.Control
          type="number"
          step="0.1"
          min="0.1"
          max="0.9"
          value={options.batchSizeFraction}
          onChange={(e) => handleChange('batchSizeFraction', parseFloat(e.target.value))}
          disabled={disabled}
          style={{ fontFamily: 'Poppins-Regular' }}
        />
        <Form.Text className="text-muted" style={{ fontFamily: 'Poppins-Regular', fontSize: '12px' }}>
          Default: 0.4 (40% of dataset per batch)
        </Form.Text>
      </Form.Group>

      <Form.Group className="mb-3">
        <Form.Check
          type="checkbox"
          id="useValidation"
          label="Use Validation Set"
          checked={options.useValidation}
          onChange={(e) => handleChange('useValidation', e.target.checked)}
          disabled={disabled}
          style={{ fontFamily: 'Poppins-Regular' }}
        />
        <Form.Text className="text-muted" style={{ fontFamily: 'Poppins-Regular', fontSize: '12px' }}>
          Split 20% of data for validation to measure model performance on unseen samples.
          Helps detect overfitting and provides more accurate accuracy metrics.
        </Form.Text>
      </Form.Group>

      <Button
        variant="outline-secondary"
        size="sm"
        onClick={handleReset}
        disabled={disabled}
        style={{ fontFamily: 'Poppins-Regular' }}
      >
        Reset to Defaults
      </Button>
    </Form>
  );
};

export default AdvancedSettings;
