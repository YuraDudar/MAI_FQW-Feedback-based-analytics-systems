export interface User {
  user_id: number
  username: string
  email: string
  role: 'admin' | 'analyst'
  is_active: boolean
  created_at: string
}

export interface TokenPair {
  access_token: string
  refresh_token: string
  token_type: string
}

export interface Product {
  product_id: number
  name: string
  source_product_id: string
  platform: 'wildberries' | 'ozon'
  user_id: number
  created_at: string
}

export interface Job {
  job_id: number
  product_id: number
  user_id: number
  job_type: 'parsing' | 'clustering' | 'auto_reply'
  status: 'pending' | 'running' | 'completed' | 'failed'
  parameters: Record<string, unknown> | null
  results_summary: Record<string, unknown> | null
  error_message: string | null
  start_time: string | null
  end_time: string | null
  created_at: string
}

export interface Review {
  review_id: string
  product_id: number
  rating: number | null
  advantages: string | null
  disadvantages: string | null
  comment: string | null
  reviewer_name: string | null
  created_date: string
  platform: string
}

export interface ReviewsListResponse {
  items: Review[]
  total: number
  page: number
  page_size: number
}

export interface Cluster {
  cluster_id: number
  sentiment_category: 'positive' | 'negative'
  bertopic_topic_id: number | null
  llm_label: string | null
  keywords: { word: string; score: number }[] | null
  review_count: number
  avg_rating: number | null
}

export interface Dashboard {
  product_id: number
  total_reviews: number
  avg_rating: number | null
  positive_count: number
  negative_count: number
  neutral_count: number
  clusters_count: number
  last_analysis: string | null
  health_score: number | null
  top_problems: { label: string; count: number; example_review_id?: string }[] | null
  top_positives: { label: string; count: number; example_review_id?: string }[] | null
}

export interface Conversation {
  conversation_id: number
  product_id: number | null
  created_at: string
}

export interface Message {
  message_id: number
  conversation_id: number
  role: 'user' | 'assistant'
  content: string
  rag_review_ids: string[] | null
  filters_applied: Record<string, unknown> | null
  created_at: string
}

export interface RAGFilters {
  date_from?: string
  date_to?: string
  stars_min?: number
  stars_max?: number
  sentiment?: 'positive' | 'negative' | 'neutral'
  gender?: 'male' | 'female' | 'unknown'
}

export interface AdminJobStats {
  pending: number
  running: number
  completed: number
  failed: number
  total: number
}
