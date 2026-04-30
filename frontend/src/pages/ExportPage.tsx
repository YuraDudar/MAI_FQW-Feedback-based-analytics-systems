import { useState } from 'react'
import { llmApi } from '../api/llm'
import { useAppStore } from '../stores/appStore'
import { Download, Loader2 } from 'lucide-react'

export default function ExportPage() {
  const { products } = useAppStore()
  const [productId, setProductId] = useState<number | ''>('')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [ratingMin, setRatingMin] = useState<string>('')
  const [ratingMax, setRatingMax] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleExport = async () => {
    if (!productId) { setError('Выберите товар'); return }
    setError('')
    setLoading(true)
    try {
      const { data } = await llmApi.exportReviews(Number(productId), {
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        rating_min: ratingMin ? Number(ratingMin) : undefined,
        rating_max: ratingMax ? Number(ratingMax) : undefined,
      })
      const url = window.URL.createObjectURL(new Blob([data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `reviews_${productId}_${Date.now()}.csv`)
      document.body.appendChild(link)
      link.click()
      link.remove()
    } catch {
      setError('Ошибка при выгрузке')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-lg space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Выгрузка отзывов</h1>
      <div className="bg-white rounded-xl border border-gray-200 p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Товар</label>
          <select
            value={productId}
            onChange={e => setProductId(e.target.value ? Number(e.target.value) : '')}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">Выберите товар...</option>
            {products.map(p => (
              <option key={p.product_id} value={p.product_id}>{p.name}</option>
            ))}
          </select>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Дата от</label>
            <input
              type="date"
              value={dateFrom}
              onChange={e => setDateFrom(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Дата до</label>
            <input
              type="date"
              value={dateTo}
              onChange={e => setDateTo(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Рейтинг от</label>
            <select value={ratingMin} onChange={e => setRatingMin(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none text-sm">
              <option value="">Любой</option>
              {[1,2,3,4,5].map(r => <option key={r} value={r}>{r} ★</option>)}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Рейтинг до</label>
            <select value={ratingMax} onChange={e => setRatingMax(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-3 py-2 focus:outline-none text-sm">
              <option value="">Любой</option>
              {[1,2,3,4,5].map(r => <option key={r} value={r}>{r} ★</option>)}
            </select>
          </div>
        </div>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button
          onClick={handleExport}
          disabled={loading}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2.5 rounded-lg font-medium transition-colors disabled:opacity-70"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
          Скачать CSV
        </button>
      </div>
    </div>
  )
}
