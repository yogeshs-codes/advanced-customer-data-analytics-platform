import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";

const API_URL = "http://localhost:8000";

function MetricCard({ title, value, subtitle }) {
  return (
    <div className="metric-card">
      <div className="metric-title">{title}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-subtitle">{subtitle}</div>
    </div>
  );
}

function App() {
  const [monitoring, setMonitoring] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  const fetchMonitoringData = async () => {
    try {
      setError("");

      const [monitoringResponse, healthResponse] = await Promise.all([
        fetch(`${API_URL}/monitoring`),
        fetch(`${API_URL}/health`),
      ]);

      if (!monitoringResponse.ok || !healthResponse.ok) {
        throw new Error("Unable to connect to the monitoring API.");
      }

      const monitoringData = await monitoringResponse.json();
      const healthData = await healthResponse.json();

      setMonitoring(monitoringData);
      setHealth(healthData);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchMonitoringData();

    const interval = setInterval(fetchMonitoringData, 10000);

    return () => clearInterval(interval);
  }, []);

  if (error && !monitoring) {
    return (
      <div className="app">
        <div className="dashboard-header">
          <div>
            <h1>Customer Demand Model Monitoring</h1>
            <p>Production prediction monitoring dashboard</p>
          </div>
        </div>

        <div className="error-panel">
          <h2>Monitoring API unavailable</h2>
          <p>{error}</p>
          <button onClick={fetchMonitoringData}>Retry connection</button>
          <p className="hint">
            Make sure the FastAPI server is running on port 8000.
          </p>
        </div>
      </div>
    );
  }

  if (!monitoring || !health) {
    return (
      <div className="loading-screen">
        <div className="loading-card">
          <h2>Loading monitoring dashboard...</h2>
          <p>Connecting to the Customer Demand Prediction API.</p>
        </div>
      </div>
    );
  }

  const predictionData = [
    {
      name: "Positive",
      value: monitoring.positive_predictions,
    },
    {
      name: "Negative",
      value: monitoring.negative_predictions,
    },
  ];

  const latencyData = [
    {
      name: "Minimum",
      latency: Number(monitoring.minimum_latency_ms || 0),
    },
    {
      name: "Average",
      latency: Number(monitoring.average_latency_ms || 0),
    },
    {
      name: "Maximum",
      latency: Number(monitoring.maximum_latency_ms || 0),
    },
  ];

  const formatPercent = (value) =>
    `${(Number(value || 0) * 100).toFixed(1)}%`;

  const formatMs = (value) =>
    `${Number(value || 0).toFixed(2)} ms`;

  const statusHealthy =
    health.status === "healthy" &&
    !monitoring.latency_anomaly &&
    !monitoring.prediction_distribution_drift;

  return (
    <div className="app">
      <header className="dashboard-header">
        <div>
          <div className="eyebrow">PHASE 3 • MODEL MONITORING</div>
          <h1>Customer Demand Model Monitoring</h1>
          <p>
            Real-time operational view of prediction activity, latency,
            probability, and model health.
          </p>
        </div>

        <div className="header-actions">
          <div className={`status-pill ${statusHealthy ? "healthy" : "warning"}`}>
            <span className="status-dot"></span>
            {statusHealthy ? "System Healthy" : "Attention Required"}
          </div>

          <button className="refresh-button" onClick={fetchMonitoringData}>
            Refresh
          </button>
        </div>
      </header>

      {error && (
        <div className="warning-banner">
          {error}
        </div>
      )}

      <section className="model-info">
        <div>
          <span>Model</span>
          <strong>{health.model_version}</strong>
        </div>

        <div>
          <span>Algorithm</span>
          <strong>{health.model}</strong>
        </div>

        <div>
          <span>Features</span>
          <strong>{health.feature_count}</strong>
        </div>

        <div>
          <span>API Status</span>
          <strong>{health.status}</strong>
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard
          title="Total Predictions"
          value={monitoring.total_predictions}
          subtitle="Predictions processed"
        />

        <MetricCard
          title="Positive Predictions"
          value={monitoring.positive_predictions}
          subtitle={formatPercent(monitoring.positive_prediction_rate)}
        />

        <MetricCard
          title="Negative Predictions"
          value={monitoring.negative_predictions}
          subtitle={formatPercent(monitoring.negative_prediction_rate)}
        />

        <MetricCard
          title="Average Latency"
          value={formatMs(monitoring.average_latency_ms)}
          subtitle="Prediction response time"
        />
      </section>

      <section className="charts-grid">
        <div className="chart-card">
          <div className="card-header">
            <div>
              <h2>Prediction Distribution</h2>
              <p>Current prediction class distribution</p>
            </div>
          </div>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={predictionData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={105}
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  <Cell />
                  <Cell />
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-card">
          <div className="card-header">
            <div>
              <h2>Latency Statistics</h2>
              <p>Minimum, average, and maximum inference latency</p>
            </div>
          </div>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="latency" name="Latency (ms)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="status-grid">
        <div className={`status-card ${monitoring.latency_anomaly ? "alert" : "ok"}`}>
          <div className="status-card-title">Latency Anomaly</div>
          <div className="status-card-value">
            {monitoring.latency_anomaly ? "Detected" : "Normal"}
          </div>
          <p>
            {monitoring.latency_anomaly
              ? "Prediction latency requires investigation."
              : "No latency anomaly detected."}
          </p>
        </div>

        <div
          className={`status-card ${
            monitoring.prediction_distribution_drift ? "alert" : "ok"
          }`}
        >
          <div className="status-card-title">
            Prediction Distribution Drift
          </div>

          <div className="status-card-value">
            {monitoring.prediction_distribution_drift
              ? "Detected"
              : "Normal"}
          </div>

          <p>
            {monitoring.prediction_distribution_drift
              ? "Prediction distribution has changed."
              : "No prediction distribution drift detected."}
          </p>
        </div>

        <div className="status-card ok">
          <div className="status-card-title">
            Average Purchase Probability
          </div>

          <div className="status-card-value">
            {formatPercent(
              monitoring.average_probability_future_purchase
            )}
          </div>

          <p>Average probability of future purchase.</p>
        </div>
      </section>

      <section className="recent-card">
        <div className="card-header">
          <div>
            <h2>Recent Predictions</h2>
            <p>
              Latest records received by the monitoring service
            </p>
          </div>

          <span className="record-count">
            {monitoring.recent_prediction_count} records
          </span>
        </div>

        {monitoring.recent_predictions.length === 0 ? (
          <div className="empty-state">
            No prediction records available.
          </div>
        ) : (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                  <th>Latency</th>
                  <th>Model</th>
                </tr>
              </thead>

              <tbody>
                {monitoring.recent_predictions.map((record, index) => (
                  <tr key={`${record.timestamp}-${index}`}>
                    <td>
                      {new Date(record.timestamp).toLocaleString()}
                    </td>

                    <td>
                      <span
                        className={`prediction-badge ${
                          record.prediction === 1
                            ? "positive"
                            : "negative"
                        }`}
                      >
                        {record.prediction === 1
                          ? "Future Purchase"
                          : "No Future Purchase"}
                      </span>
                    </td>

                    <td>
                      {formatPercent(
                        record.probability_future_purchase
                      )}
                    </td>

                    <td>
                      {formatMs(record.latency_ms)}
                    </td>

                    <td>{record.model}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <footer>
        <span>
          Customer Demand Analysis • Model Monitoring
        </span>

        <span>
          Last updated:{" "}
          {lastUpdated
            ? lastUpdated.toLocaleTimeString()
            : "Loading..."}
        </span>
      </footer>
    </div>
  );
}

export default App;