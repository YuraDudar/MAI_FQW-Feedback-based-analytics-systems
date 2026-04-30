import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { productsApi } from '../api/products'
import type { Review } from '../types'
import { Loader2, Star, ChevronLeft, ChevronRight } from 'lucide-react'

export default function ReviewsPage() {
  const { productId } = useParams<{ productId: string }>()
  const [reviews, setReviews] = useState<Review[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [ratingMin, setRatingMin] = useState<string>('')
  const [ratingMax, setRatingMax] = useState<string>('')
  const PAGE_SIZE = 20

  useEffect(() => {
    if (productId) loadReviews()
  }, [productId, page, ratingMin, ratingMax])

  const loadReviews = async () => {
    setLoading(true)
    try {
      const { data } = await productsApi.getReviews(Number(productId), {
        page,
        page_size: PAGE_SIZE,
        rating_min: ratingMin ? Number(ratingMin) : undefined,
        rating_max: ratingMax ? Number(ratingMax) : undefined,
      })
      setReviews(data.items)
      setTotal(data.total)
    } finally {
      setLoading(false)
    }
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)

  const StarRating = ({ rating }: { rating: number | null }) => (
    <div className="flex gap-0.5">
      {[1,2,3,4,5].map(s => (
        <Star
          key={s}
          className={`w-3.5 h-3.5 ${s <= (rating ?? 0) ? 'text-yellow-400 fill-yellow-400' : 'text-gray-300'}`}
        />
      ))}
    </div>
  )

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold text-gray-900">
          Отзывы <span className="text-base font-normal text-gray-400">({total})</span>
        </h1>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-sm">
            <span className="text-gray-500">Рейтинг:</span>
            <select
              value={ratingMin}
              onChange={e => { setRatingMin(e.target.value); setPage(1) }}
              className="border border-gray-200 rounded-lg px-2 py-1 text-sm focus:outline-none"
            >
              <option value="">от 1</option>
              {[1,2,3,4,5].map(r => <option key={r} value={r}>от {r}</option>)}
            </select>
            <select
              value={ratingMax}
              onChange={e => { setRatingMax(e.target.value); setPage(1) }}
              className="border border-gray-200 rounded-lg px-2 py-1 text-sm focus:outline-none"
            >
              <option value="">до 5</option>
              {[1,2,3,4,5].map(r => <option key={r} value={r}>до {r}</option>)}
            </select>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center py-16">
          <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
        </div>
      ) : (
        <div className="space-y-3">
          {reviews.map(review => (
            <div key={review.review_id} className="bg-white rounded-xl border border-gray-100 p-5 shadow-sm">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <span className="font-medium text-gray-900 text-sm">{review.reviewer_name || 'Аноним'}</span>
                  <span className="text-gray-400 text-xs ml-2">
                    {new Date(review.created_date).toLocaleDateString('ru-RU')}
                  </span>
                </div>
                <StarRating rating={review.rating} />
              </div>
              {review.advantages && (
                <p className="text-sm text-green-700 mb-1">
                  <span className="font-medium">✓ Достоинства:</span> {review.advantages}
                </p>
              )}
              {review.disadvantages && (
                <p className="text-sm text-red-700 mb-1">
                  <span className="font-medium">✕ Недостатки:</span> {review.disadvantages}
                </p>
              )}
              {review.comment && (
                <p className="text-sm text-gray-700">
                  <span className="font-medium">Комментарий:</span> {review.comment}
                </p>
              )}
            </div>
          ))}
        </div>
      )}

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-3">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="p-2 rounded-lg border border-gray-200 disabled:opacity-40 hover:bg-gray-50"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-sm text-gray-600">
            Стр. {page} из {totalPages}
          </span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="p-2 rounded-lg border border-gray-200 disabled:opacity-40 hover:bg-gray-50"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  )
}
