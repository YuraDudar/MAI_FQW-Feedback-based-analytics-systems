import apiClient from './client'
import type { Conversation, Message, RAGFilters } from '../types'

export const llmApi = {
  createConversation: (productId?: number) =>
    apiClient.post<Conversation>('/llm/conversations', { product_id: productId }),

  listConversations: () => apiClient.get<Conversation[]>('/llm/conversations'),

  getMessages: (conversationId: number) =>
    apiClient.get<Message[]>(`/llm/conversations/${conversationId}/messages`),

  sendMessage: (conversationId: number, content: string, filters?: RAGFilters, topK?: number) =>
    apiClient.post<Message>(`/llm/conversations/${conversationId}/messages`, {
      content,
      filters,
      top_k: topK ?? 10,
    }),

  exportReviews: (productId: number, params?: {
    date_from?: string; date_to?: string
    rating_min?: number; rating_max?: number
  }) =>
    apiClient.get(`/export/reviews/${productId}/csv`, {
      params,
      responseType: 'blob',
    }),
}
