import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { productsApi } from '../api/products'
import { useAppStore } from '../stores/appStore'
import type { Product, Dashboard } from '../types'
import {
  PlusCircle, Star, TrendingUp, TrendingDown,
  Loader2, AlertCircle, Play, Network,
} from 'lucide-react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'

const COLORS = ['#22c55e', '#ef4444', '#94a3b8']

export default function DashboardPage() {
  const { products, setProducts, addProduct } = useAppStore()
  const [dashboards, setDashboards] = useState<Record<number, Dashboard>>({})
  const [loading, setLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)
  const [addForm, setAddForm] = useState({ name: '', sku: '', platform: 'wildberries' })
  const [addLoading, setAddLoading] = useState(false)
  const [addError, setAddError] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    loadProducts()
  }, [])

  const loadProducts = async () => {
    setLoading(true)
    try {
      const { data } = await productsApi.list()
      setProducts(data)
      await Promise.all(data.map(p => loadDashboard(p.product_id)))
    } finally {
      setLoading(false)
    }
  }

  const loadDashboard = async (productId: number) => {
    try {
      const { data } = await productsApi.getDashboard(productId)
      setDashboards(prev => ({ ...prev, [productId]: data }))
    } catch {}
  }

  const handleAddProduct = async (e: React.FormEvent) => {
    e.preventDefault()
    setAddError('')
    setAddLoading(true)
    try {
      const { data } = await productsApi.create(addForm.name, addForm.sku, addForm.platform)
      addProduct(data)
      setShowAddModal(false)
      setAddForm({ name: '', sku: '', platform: 'wildberries' })
      await productsApi.startParsing(data.product_id)
    } catch (err: any) {
      setAddError(err.response?.data?.detail || 'Ошибка добавления товара')
    } finally {
      setAddLoading(false)
    }
  }

  const handleStartParsing = async (productId: number) => {
    try {
      await productsApi.startParsing(productId)
    } catch {}
  }

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
        <h1 className="text-2xl font-bold text-gray-900">Дашборд</h1>
        <button
          onClick={() => setShowAddModal(true)}
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          <PlusCircle className="w-4 h-4" />
          Добавить товар
        </button>
      </div>

      {products.length === 0 ? (
        <div className="text-center py-16 bg-white rounded-xl border border-dashed border-gray-300">
          <AlertCircle className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-600 mb-2">Нет товаров</h3>
          <p className="text-sm text-gray-500 mb-4">Добавьте первый товар для анализа отзывов</p>
          <button
            onClick={() => setShowAddModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium"
          >
            Добавить товар
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {products.map(product => {
            const dash = dashboards[product.product_id]
            const sentimentData = dash ? [
              { name: 'Позит.', value: dash.positive_count, color: '#22c55e' },
              { name: 'Негат.', value: dash.negative_count, color: '#ef4444' },
              { name: 'Нейтр.', value: dash.neutral_count, color: '#94a3b8' },
            ] : []

            return (
              <div key={product.product_id} className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">{product.name}</h2>
                    <p className="text-sm text-gray-500">
                      SKU: {product.source_product_id} · {product.platform === 'wildberries' ? 'Wildberries' : 'Ozon'}
                    </p>
                  </div>
                  {dash?.health_score != null && (
                    <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                      dash.health_score >= 7 ? 'bg-green-100 text-green-700' :
                      dash.health_score >= 4 ? 'bg-yellow-100 text-yellow-700' :
                      'bg-red-100 text-red-700'
                    }`}>
                      {dash.health_score.toFixed(1)} / 10
                    </div>
                  )}
                </div>

                {dash && (
                  <>
                    <div className="grid grid-cols-3 gap-3 mb-4">
                      <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold text-gray-900">{dash.total_reviews}</p>
                        <p className="text-xs text-gray-500 mt-1">Всего отзывов</p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold text-yellow-600">
                          {dash.avg_rating != null ? dash.avg_rating.toFixed(1) : '—'}
                        </p>
                        <p className="text-xs text-gray-500 mt-1">Ср. рейтинг</p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3 text-center">
                        <p className="text-2xl font-bold text-blue-600">{dash.clusters_count}</p>
                        <p className="text-xs text-gray-500 mt-1">Кластеров</p>
                      </div>
                    </div>

                    {sentimentData.some(d => d.value > 0) && (
                      <div className="h-32 mb-4">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={sentimentData} barSize={32}>
                            <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                            <YAxis hide />
                            <Tooltip />
                            <Bar dataKey="value">
                              {sentimentData.map((d, i) => (
                                <Cell key={i} fill={d.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </>
                )}

                <div className="flex gap-2 mt-4">
                  <button
                    onClick={() => navigate(`/reviews/${product.product_id}`)}
                    className="flex-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-1"
                  >
                    <Star className="w-4 h-4" />
                    Отзывы
                  </button>
                  <button
                    onClick={() => navigate(`/clusters/${product.product_id}`)}
                    className="flex-1 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-1"
                  >
                    <Network className="w-4 h-4" />
                    Кластеры
                  </button>
                  <button
                    onClick={() => handleStartParsing(product.product_id)}
                    className="flex-1 text-sm bg-blue-50 hover:bg-blue-100 text-blue-700 px-3 py-2 rounded-lg font-medium transition-colors flex items-center justify-center gap-1"
                  >
                    <Play className="w-4 h-4" />
                    Обновить
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Добавить товар</h2>
            <form onSubmit={handleAddProduct} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Название товара</label>
                <input
                  type="text"
                  value={addForm.name}
                  onChange={e => setAddForm(prev => ({ ...prev, name: e.target.value }))}
                  required
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Название товара"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Артикул (nmId для WB)</label>
                <input
                  type="text"
                  value={addForm.sku}
                  onChange={e => setAddForm(prev => ({ ...prev, sku: e.target.value }))}
                  required
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="123456789"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Платформа</label>
                <select
                  value={addForm.platform}
                  onChange={e => setAddForm(prev => ({ ...prev, platform: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="wildberries">Wildberries</option>
                  <option value="ozon">Ozon</option>
                </select>
              </div>
              {addError && (
                <div className="bg-red-50 text-red-700 text-sm px-3 py-2 rounded-lg border border-red-200">
                  {addError}
                </div>
              )}
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Отмена
                </button>
                <button
                  type="submit"
                  disabled={addLoading}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center justify-center gap-2 disabled:opacity-70"
                >
                  {addLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                  Добавить и запустить
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
