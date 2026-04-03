import React from 'react';
import { ImageMapProvider } from './contexts/ImageMapContext';
import { WebcamProvider } from './contexts/WebcamContext';
import { ImageClassifierProvider } from './contexts/ImageClassifierContext';
import TeachableMachineLayout from './components/TeachableMachineLayout';

function App() {
  return (
    <div className="App" style={{ backgroundColor: '#f8f9fa', minHeight: '100vh' }}>
      <ImageMapProvider>
        <WebcamProvider>
          <ImageClassifierProvider>
            <TeachableMachineLayout />
          </ImageClassifierProvider>
        </WebcamProvider>
      </ImageMapProvider>
    </div>
  );
}

export default App;
