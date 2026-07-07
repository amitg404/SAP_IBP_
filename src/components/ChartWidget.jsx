// ChartWidget — renders Recharts charts from the optional `chart` payload in Billy's response.
// Props: { chart: { chart_type, title, x_key, y_key, data } }
// Supports: line (trend), bar (comparison), pie (breakdown)

import {
  ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  BarChart, Bar,
  PieChart, Pie, Cell,
} from 'recharts';

// ── Palette — harmonises with index.css dark/glassmorphism theme ──────────────
const COLORS = ['#818cf8', '#34d399', '#fb923c', '#f472b6', '#60a5fa', '#a78bfa'];

// Format large numbers compactly on axes (12000 → 12k)
const fmtQty = (v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v;

// Tooltip content styled to match the dark theme
const DarkTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(15,23,42,0.95)',
      border: '1px solid rgba(129,140,248,0.3)',
      borderRadius: '10px',
      padding: '10px 14px',
      fontSize: '13px',
      color: '#e2e8f0',
    }}>
      {label && <p style={{ margin: '0 0 6px', color: '#94a3b8', fontWeight: 600 }}>{label}</p>}
      {payload.map((entry, i) => (
        <p key={i} style={{ margin: '2px 0', color: entry.color }}>
          {entry.name}: <strong>{Number(entry.value).toLocaleString()}</strong>
        </p>
      ))}
    </div>
  );
};

// ── Line chart — inventory over time ─────────────────────────────────────────
function TrendLine({ data, x_key, y_key }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" />
        <XAxis
          dataKey={x_key}
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
        />
        <YAxis
          tickFormatter={fmtQty}
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          width={40}
        />
        <Tooltip content={<DarkTooltip />} />
        <Line
          type="monotone"
          dataKey={y_key}
          name="Inventory"
          stroke="#818cf8"
          strokeWidth={2.5}
          dot={{ fill: '#818cf8', r: 3 }}
          activeDot={{ r: 5, fill: '#a5b4fc' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ── Bar chart — region/product comparison ────────────────────────────────────
function CompareBar({ data, x_key, y_key }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.07)" vertical={false} />
        <XAxis
          dataKey={x_key}
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          tickLine={false}
          axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
        />
        <YAxis
          tickFormatter={fmtQty}
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          width={40}
        />
        <Tooltip content={<DarkTooltip />} />
        <Bar dataKey={y_key} name="Inventory" radius={[6, 6, 0, 0]}>
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Pie chart — aggregate by region/product breakdown ────────────────────────
const RADIAN = Math.PI / 180;
const renderLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) => {
  const r  = innerRadius + (outerRadius - innerRadius) * 0.5;
  const x  = cx + r * Math.cos(-midAngle * RADIAN);
  const y  = cy + r * Math.sin(-midAngle * RADIAN);
  return percent > 0.05 ? (
    <text x={x} y={y} fill="white" textAnchor="middle" dominantBaseline="central" fontSize={11}>
      {`${(percent * 100).toFixed(0)}%`}
    </text>
  ) : null;
};

function BreakdownPie({ data, x_key, y_key }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie
          data={data}
          dataKey={y_key}
          nameKey={x_key}
          cx="50%"
          cy="50%"
          outerRadius={90}
          labelLine={false}
          label={renderLabel}
        >
          {data.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          content={<DarkTooltip />}
          formatter={(v, name) => [Number(v).toLocaleString(), name]}
        />
        <Legend
          formatter={(value) => <span style={{ color: '#94a3b8', fontSize: 12 }}>{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

// ── Main export ───────────────────────────────────────────────────────────────
export default function ChartWidget({ chart }) {
  if (!chart?.data?.length) return null;
  const { chart_type, title, x_key, y_key, data } = chart;

  return (
    <div className="chart-widget" role="img" aria-label={title}>
      {title && <p className="chart-title">{title}</p>}
      {chart_type === 'line' && <TrendLine  data={data} x_key={x_key} y_key={y_key} />}
      {chart_type === 'bar'  && <CompareBar data={data} x_key={x_key} y_key={y_key} />}
      {chart_type === 'pie'  && <BreakdownPie data={data} x_key={x_key} y_key={y_key} />}
    </div>
  );
}
