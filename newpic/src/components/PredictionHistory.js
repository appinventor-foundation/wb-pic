import React, { useMemo } from 'react';
import Button from 'react-bootstrap/Button';
import ProgressBar from 'react-bootstrap/ProgressBar';
import { useImageMap } from '../contexts/ImageMapContext';

const formatTime = (time) =>
  new Date(time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

const THUMB_SIZE = 140;

const ResultCard = ({ entry }) => {
  const confidence = Math.round(entry.predictions[0].confidence * 100);
  const title = [
    `${entry.runLabel} (run ${entry.runNumber}) · ${formatTime(entry.time)}`,
    ...entry.predictions.map(p => `${p.label}: ${Math.round(p.confidence * 100)}%`)
  ].join('\n');

  return (
    <div title={title} style={{ width: `${THUMB_SIZE}px`, flexShrink: 0 }}>
      {entry.thumb ? (
        <img
          src={entry.thumb}
          alt=""
          style={{ width: `${THUMB_SIZE}px`, height: `${THUMB_SIZE}px`, borderRadius: '10px', objectFit: 'cover', display: 'block' }}
        />
      ) : (
        <div style={{ width: `${THUMB_SIZE}px`, height: `${THUMB_SIZE}px`, borderRadius: '10px', backgroundColor: '#eee' }} />
      )}
      <ProgressBar
        now={confidence}
        label={`${confidence}%`}
        variant="primary"
        style={{ height: '20px', marginTop: '6px', fontSize: '13px', borderRadius: '6px' }}
      />
    </div>
  );
};

const PredictionHistory = () => {
  const { history, clearHistory } = useImageMap();

  // Group every inference by its top predicted label; newest first within a group
  const groups = useMemo(() => {
    const byLabel = {};
    history.forEach(run => {
      run.entries.forEach(entry => {
        const label = entry.predictions[0].label;
        (byLabel[label] = byLabel[label] || []).push({ ...entry, runNumber: run.number, runLabel: run.label });
      });
    });
    return Object.keys(byLabel)
      .sort()
      .map(label => ({ label, entries: byLabel[label].sort((a, b) => b.time - a.time) }));
  }, [history]);

  if (groups.length === 0) return null;

  return (
    <div style={{
      backgroundColor: 'white',
      borderRadius: '12px',
      padding: '24px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      marginTop: '24px'
    }}>
      <div style={{ position: 'relative', textAlign: 'center', marginBottom: '24px' }}>
        <h3 style={{ fontSize: '24px', fontWeight: '500', margin: 0, fontFamily: 'Poppins-Regular', color: '#1a1a1d' }}>
          Test Results
        </h3>
        <Button
          variant="outline-secondary"
          size="sm"
          onClick={clearHistory}
          style={{ position: 'absolute', right: 0, top: '50%', transform: 'translateY(-50%)' }}
        >
          Clear history
        </Button>
      </div>

      {groups.map(({ label, entries }) => (
        <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '24px', marginBottom: '20px' }}>
          <div style={{
            width: '140px',
            flexShrink: 0,
            padding: '14px 8px',
            textAlign: 'center',
            backgroundColor: '#d1ecf1',
            border: '1px solid #bee5eb',
            borderRadius: '6px',
            fontFamily: 'Poppins-Regular',
            wordBreak: 'break-word'
          }}>
            {label}
          </div>
          {/* One row per label: grows to the right and scrolls instead of wrapping */}
          <div style={{ display: 'flex', flexWrap: 'nowrap', gap: '16px', overflowX: 'auto', minWidth: 0, flex: 1, paddingBottom: '8px' }}>
            {entries.map(entry => (
              <ResultCard key={entry.id} entry={entry} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default PredictionHistory;
