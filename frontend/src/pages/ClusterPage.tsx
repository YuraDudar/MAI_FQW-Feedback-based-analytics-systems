import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { productsApi } from '../api/products'
import type { Cluster } from '../types'
import { Loader2 } from 'lucide-react'
import Plot from 'react-plotly.js'

interface ClusterPoint {
  x: number
  y: number
  cluster_id: number
  label: string
  sentiment: string
  review_count: number
}

export default function ClusterPage() {
  const { productId } = useParams<{ productId: string }>()
  const [clusters, setClusters] = useState<Cluster[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'positive' | 'negative'>('all')

  useEffect(() => {
    if (productId) loadClusters(Number(productId))
  }, [productId])

  const loadClusters = async (id: number) => {
    setLoading(true)
    try {
      const { data } = await productsApi.getClusters(id)
      setClusters(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const filtered = clusters.filter(c =>
    filter === 'all' ? true : c.sentiment_category === filter
  )

  const posClusters = clusters.filter(c => c.sentiment_category === 'positive')
  const negClusters = clusters.filter(c => c.sentiment_category === 'negative')

  const makePlotData = (list: Cluster[], color: string) => ({
    x: list.map((_, i) => Math.cos((2 * Math.PI * i) / Math.max(list.length, 1))),
    y: list.map((_, i) => Math.sin((2 * Math.PI * i) / Math.max(list.length, 1))),
    mode: 'markers+text' as const,
    type: 'scatter' as const,
    marker: {
      size: list.map(c => Math.max(10, Math.min(50, c.review_count / 5))),
      color,
      opacity: 0.7,
    },
    text: list.map(c => c.llm_label || `Кластер ${c.cluster_id}`),
    textposition: 'top center' as const,
    hovertemplate: list.map(c =>
      `<b>${c.llm_label || `Кластер ${c.cluster_id}`}</b><br>Отзывов: ${c.review_count}<br>Рейтинг: ${c.avg_rating?.toFixed(1) ?? '—'}<extra></extra>`
    ),
    name: color === '#22c55e' ? 'Положительные' : 'Отрицательные',
  })

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Кластеры отзывов</h1>
        <div className="flex gap-2">
          {(['all', 'positive', 'negative'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                filter === f
                  ? f === 'positive' ? 'bg-green-100 text-green-700' :
                    f === 'negative' ? 'bg-red-100 text-red-700' :
                    'bg-blue-100 text-blue-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {f === 'all' ? 'Все' : f === 'positive' ? 'Положительные' : 'Отрицательные'}
            </button>
          ))}
        </div>
      </div>

      {clusters.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-xl border border-dashed border-gray-300">
          <p className="text-gray-500">Кластеры ещё не сформированы. Запустите анализ на дашборде.</p>
        </div>
      ) : (
        <>
          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
            <h2 className="text-base font-semibold text-gray-700 mb-3">Интерактивный граф кластеров</h2>
            <Plot
              data={[
                ...(filter !== 'negative' ? [makePlotData(posClusters, '#22c55e')] : []),
                ...(filter !== 'positive' ? [makePlotData(negClusters, '#ef4444')] : []),
              ]}
              layout={{
                autosize: true,
                height: 480,
                showlegend: true,
                hovermode: 'closest',
                margin: { l: 40, r: 40, t: 20, b: 40 },
                xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { showgrid: false, zeroline: false, showticklabels: false },
              }}
              style={{ width: '100%' }}
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filtered.map(cluster => (
              <div
                key={cluster.cluster_id}
                className={`bg-white rounded-xl border p-4 ${
                  cluster.sentiment_category === 'positive'
                    ? 'border-green-200'
                    : 'border-red-200'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-medium text-gray-900 text-sm leading-snug">
                    {cluster.llm_label || `Кластер ${cluster.cluster_id}`}
                  </h3>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium shrink-0 ml-2 ${
                    cluster.sentiment_category === 'positive'
                      ? 'bg-green-100 text-green-700'
                      : 'bg-red-100 text-red-700'
                  }`}>
                    {cluster.sentiment_category === 'positive' ? '+ Позит.' : '− Негат.'}
                  </span>
                </div>
                <div className="flex gap-3 text-xs text-gray-500 mb-3">
                  <span>{cluster.review_count} отзывов</span>
                  {cluster.avg_rating != null && <span>★ {cluster.avg_rating.toFixed(1)}</span>}
                </div>
                {cluster.keywords && cluster.keywords.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {cluster.keywords.slice(0, 6).map((kw, i) => (
                      <span key={i} className="bg-gray-100 text-gray-600 text-xs px-2 py-0.5 rounded-full">
                        {kw.word}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
