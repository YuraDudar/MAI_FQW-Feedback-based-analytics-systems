import { useState, useEffect, useRef } from 'react'
import { llmApi } from '../api/llm'
import { useAppStore } from '../stores/appStore'
import type { Conversation, Message, RAGFilters } from '../types'
import { Send, Loader2, Bot, User2, Filter, Plus } from 'lucide-react'

export default function ChatPage() {
  const { products } = useAppStore()
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [selectedConv, setSelectedConv] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState<number | undefined>(undefined)
  const [topK, setTopK] = useState(10)
  const [showFilters, setShowFilters] = useState(false)
  const [filters, setFilters] = useState<RAGFilters>({})
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    loadConversations()
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const loadConversations = async () => {
    const { data } = await llmApi.listConversations()
    setConversations(data)
    if (data.length > 0) {
      await openConversation(data[0].conversation_id)
    }
  }

  const openConversation = async (id: number) => {
    setSelectedConv(id)
    const { data } = await llmApi.getMessages(id)
    setMessages(data)
  }

  const createConversation = async () => {
    const { data } = await llmApi.createConversation(selectedProduct)
    setConversations(prev => [data, ...prev])
    setSelectedConv(data.conversation_id)
    setMessages([])
  }

  const sendMessage = async () => {
    if (!input.trim() || !selectedConv || sending) return
    const text = input.trim()
    setInput('')
    setSending(true)

    const userMsg: Message = {
      message_id: Date.now(),
      conversation_id: selectedConv,
      role: 'user',
      content: text,
      rag_review_ids: null,
      filters_applied: null,
      created_at: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg])

    try {
      const { data } = await llmApi.sendMessage(selectedConv, text, Object.keys(filters).length > 0 ? filters : undefined, topK)
      setMessages(prev => [...prev, data])
    } catch (err: any) {
      const errMsg: Message = {
        message_id: Date.now() + 1,
        conversation_id: selectedConv,
        role: 'assistant',
        content: 'Ошибка: не удалось получить ответ. Проверьте, что товар проиндексирован.',
        rag_review_ids: null,
        filters_applied: null,
        created_at: new Date().toISOString(),
      }
      setMessages(prev => [...prev, errMsg])
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex h-[calc(100vh-7rem)] gap-4">
      <div className="w-64 bg-white rounded-xl border border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-100">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-semibold text-gray-900 text-sm">Беседы</h2>
            <button
              onClick={createConversation}
              className="p-1 hover:bg-gray-100 rounded-lg text-gray-500 transition-colors"
              title="Новая беседа"
            >
              <Plus className="w-4 h-4" />
            </button>
          </div>
          <select
            value={selectedProduct ?? ''}
            onChange={e => setSelectedProduct(e.target.value ? Number(e.target.value) : undefined)}
            className="w-full text-xs border border-gray-200 rounded-lg px-2 py-1.5 focus:outline-none"
          >
            <option value="">Без привязки к товару</option>
            {products.map(p => (
              <option key={p.product_id} value={p.product_id}>{p.name}</option>
            ))}
          </select>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {conversations.map(conv => (
            <button
              key={conv.conversation_id}
              onClick={() => openConversation(conv.conversation_id)}
              className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                selectedConv === conv.conversation_id
                  ? 'bg-blue-50 text-blue-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <p className="font-medium truncate">
                Беседа #{conv.conversation_id}
              </p>
              <p className="text-xs text-gray-400 mt-0.5">
                {new Date(conv.created_at).toLocaleDateString('ru-RU')}
              </p>
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 bg-white rounded-xl border border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-100 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-blue-600" />
            <h2 className="font-semibold text-gray-900">Чат с ИИ-ассистентом</h2>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <span>top-k:</span>
              <input
                type="number"
                value={topK}
                onChange={e => setTopK(Math.max(5, Math.min(40, Number(e.target.value))))}
                className="w-12 border border-gray-200 rounded px-1 py-0.5 text-center"
                min={5} max={40}
              />
            </div>
            <button
              onClick={() => setShowFilters(v => !v)}
              className={`flex items-center gap-1 text-xs px-2 py-1 rounded-lg transition-colors ${
                showFilters ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <Filter className="w-3 h-3" />
              Фильтры
            </button>
          </div>
        </div>

        {showFilters && (
          <div className="px-4 py-3 bg-gray-50 border-b border-gray-100 grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Тональность</label>
              <select
                value={filters.sentiment ?? ''}
                onChange={e => setFilters(prev => ({ ...prev, sentiment: e.target.value as any || undefined }))}
                className="w-full text-xs border border-gray-200 rounded px-2 py-1"
              >
                <option value="">Любая</option>
                <option value="positive">Позитив</option>
                <option value="negative">Негатив</option>
                <option value="neutral">Нейтральный</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Пол</label>
              <select
                value={filters.gender ?? ''}
                onChange={e => setFilters(prev => ({ ...prev, gender: e.target.value as any || undefined }))}
                className="w-full text-xs border border-gray-200 rounded px-2 py-1"
              >
                <option value="">Любой</option>
                <option value="male">Мужской</option>
                <option value="female">Женский</option>
                <option value="unknown">Не определён</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Рейтинг от</label>
              <input
                type="number"
                min={1} max={5}
                value={filters.stars_min ?? ''}
                onChange={e => setFilters(prev => ({ ...prev, stars_min: e.target.value ? Number(e.target.value) : undefined }))}
                className="w-full text-xs border border-gray-200 rounded px-2 py-1"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Рейтинг до</label>
              <input
                type="number"
                min={1} max={5}
                value={filters.stars_max ?? ''}
                onChange={e => setFilters(prev => ({ ...prev, stars_max: e.target.value ? Number(e.target.value) : undefined }))}
                className="w-full text-xs border border-gray-200 rounded px-2 py-1"
              />
            </div>
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-gray-400 text-sm py-8">
              Задайте вопрос об отзывах товара
            </div>
          )}
          {messages.map(msg => (
            <div key={msg.message_id} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4 text-blue-600" />
                </div>
              )}
              <div className={`max-w-2xl rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white rounded-br-sm'
                  : 'bg-gray-100 text-gray-800 rounded-bl-sm'
              }`}>
                <p className="whitespace-pre-wrap">{msg.content}</p>
                {msg.rag_review_ids && msg.rag_review_ids.length > 0 && (
                  <p className="text-xs mt-2 opacity-60">
                    Источники: {msg.rag_review_ids.length} отзывов
                  </p>
                )}
              </div>
              {msg.role === 'user' && (
                <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center shrink-0">
                  <User2 className="w-4 h-4 text-gray-600" />
                </div>
              )}
            </div>
          ))}
          {sending && (
            <div className="flex gap-3 justify-start">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                <Bot className="w-4 h-4 text-blue-600" />
              </div>
              <div className="bg-gray-100 rounded-2xl px-4 py-3">
                <Loader2 className="w-4 h-4 animate-spin text-gray-500" />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="p-4 border-t border-gray-100">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
              placeholder="Введите вопрос..."
              className="flex-1 px-4 py-2.5 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={!selectedConv || sending}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || !selectedConv || sending}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2.5 rounded-xl transition-colors disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
