import React from 'react';
import { WarningOctagon, ArrowClockwise } from '@phosphor-icons/react';
import './ErrorBoundary.css';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ errorInfo });
    console.error('[ErrorBoundary] Caught error:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-boundary-inner">
            <div className="error-boundary-icon">
              <WarningOctagon size={40} weight="duotone" />
            </div>
            <h1 className="error-boundary-title">Something went wrong</h1>
            <p className="error-boundary-desc">
              An unexpected error occurred in the monitoring dashboard.
              Patient data streams are unaffected.
            </p>
            {this.state.error && (
              <div className="error-boundary-detail">
                <code>{this.state.error.toString()}</code>
              </div>
            )}
            <div className="error-boundary-actions">
              <button
                className="error-boundary-btn error-boundary-btn-primary"
                onClick={this.handleReset}
              >
                <ArrowClockwise size={14} weight="bold" />
                Try Again
              </button>
              <button
                className="error-boundary-btn error-boundary-btn-secondary"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
