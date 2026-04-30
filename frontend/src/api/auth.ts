import apiClient from './client'
import type { User, TokenPair } from '../types'

export const authApi = {
  register: (username: string, email: string, password: string) =>
    apiClient.post<User>('/auth/register', { username, email, password }),

  login: (username: string, password: string) =>
    apiClient.post<TokenPair>('/auth/login', { username, password }),

  refresh: (refreshToken: string) =>
    apiClient.post<TokenPair>('/auth/refresh', { refresh_token: refreshToken }),

  me: () => apiClient.get<User>('/auth/me'),
}
