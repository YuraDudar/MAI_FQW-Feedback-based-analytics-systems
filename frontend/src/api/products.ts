import apiClient from './client'
import type { Product, Dashboard, Job, ReviewsListResponse, Cluster } from '../types'

export const productsApi = {
  create: (name: string, source_product_id: string, platform: string) =>
    apiClient.post<Product>('/products', { name, source_product_id, platform }),

  list: () => apiClient.get<Product[]>('/products'),

  get: (id: number) => apiClient.get<Product>(`/products/${id}`),

  delete: (id: number) => apiClient.delete(`/products/${id}`),

  getDashboard: (id: number) => apiClient.get<Dashboard>(`/reviews/${id}/dashboard`),

  startParsing: (productId: number, maxReviews?: number) =>
    apiClient.post<Job>('/jobs/parse', { product_id: productId, max_reviews: maxReviews }),

  startAutoReply: (productId: number, reviewIds: string[]) =>
    apiClient.post<Job>('/jobs/auto-reply', { product_id: productId, review_ids: reviewIds }),

  listJobs: (productId?: number) =>
    apiClient.get<Job[]>('/jobs', { params: productId ? { product_id: productId } : {} }),

  getJob: (jobId: number) => apiClient.get<Job>(`/jobs/${jobId}`),

  getReviews: (productId: number, params?: {
    page?: number; page_size?: number
    rating_min?: number; rating_max?: number
    date_from?: string; date_to?: string
  }) => apiClient.get<ReviewsListResponse>(`/reviews/${productId}`, { params }),

  getClusters: (productId: number) =>
    apiClient.get<Cluster[]>(`/internal/clusters/${productId}`).catch(() =>
      apiClient.get<Cluster[]>(`/ml/clusters/${productId}`)
    ),
}
