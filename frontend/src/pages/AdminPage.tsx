import { useState, useEffect } from 'react'
import apiClient from '../api/client'
import type { AdminJobStats } from '../types'
import { Loader2, Activity, CheckCircle, XCircle, Clock, Zap } from 'lucide-react'

export default function AdminPage() {
  const [jobStats, setJobStats] = useState<AdminJobStats | null>(null)
  const [mlHealth, setMlHealth] = useState<{ status: string; code?: number } | null>(null)
  const [users, setUsers] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      const [statsRes, healthRes, usersRes] = await Promise.allSettled([
        apiClient.get<AdminJobStats>('/admin/jobs/stats'),
        apiClient.get('/admin/ml/health'),
        apiClient.get('/admin/users/list'),
      ])
      if (statsRes.status === 'fulfilled') setJobStats(statsRes.value.data)
      if (healthRes.status === 'fulfilled') setMlHealth(healthRes.value.data)
      if (usersRes.status === 'fulfilled') setUsers(usersRes.value.data)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="flex justify-center py-16"><Loader2 className="w-8 h-8 animate-spin text-blue-600" /></div>
  }

  const statCards = jobStats ? [
    { label: 'В очереди', value: jobStats.pending, icon: Clock, color: 'text-yellow-500 bg-yellow-50' },
    { label: 'Выполняется', value: jobStats.running, icon: Zap, color: 'text-blue-500 bg-blue-50' },
    { label: 'Завершено', value: jobStats.completed, icon: CheckCircle, color: 'text-green-500 bg-green-50' },
    { label: 'Ошибки', value: jobStats.failed, icon: XCircle, color: 'text-red-500 bg-red-50' },
  ] : []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Администрирование</h1>
        <button onClick={loadData} className="flex items-center gap-2 text-sm bg-gray-100 hover:bg-gray-200 px-3 py-2 rounded-lg">
          <Activity className="w-4 h-4" />
          Обновить
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {statCards.map(({ label, value, icon: Icon, color }) => (
          <div key={label} className="bg-white rounded-xl border border-gray-100 p-4 flex items-center gap-3">
            <div className={`p-2 rounded-lg ${color}`}>
              <Icon className="w-5 h-5" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900">{value}</p>
              <p className="text-xs text-gray-500">{label}</p>
            </div>
          </div>
        ))}
      </div>

      {mlHealth && (
        <div className="bg-white rounded-xl border border-gray-100 p-5">
          <h2 className="font-semibold text-gray-900 mb-3">Статус ML-сервиса</h2>
          <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
            mlHealth.status === 'ok' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
          }`}>
            {mlHealth.status === 'ok' ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
            {mlHealth.status === 'ok' ? 'Работает' : 'Недоступен'}
          </div>
        </div>
      )}

      <div className="bg-white rounded-xl border border-gray-100 p-5">
        <h2 className="font-semibold text-gray-900 mb-4">Пользователи ({users.length})</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100">
                <th className="text-left py-2 pr-4 text-gray-500 font-medium">ID</th>
                <th className="text-left py-2 pr-4 text-gray-500 font-medium">Имя</th>
                <th className="text-left py-2 pr-4 text-gray-500 font-medium">Email</th>
                <th className="text-left py-2 pr-4 text-gray-500 font-medium">Роль</th>
                <th className="text-left py-2 text-gray-500 font-medium">Статус</th>
              </tr>
            </thead>
            <tbody>
              {users.map(u => (
                <tr key={u.user_id} className="border-b border-gray-50 hover:bg-gray-50">
                  <td className="py-2 pr-4 text-gray-500">{u.user_id}</td>
                  <td className="py-2 pr-4 font-medium text-gray-900">{u.username}</td>
                  <td className="py-2 pr-4 text-gray-600">{u.email}</td>
                  <td className="py-2 pr-4">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      u.role === 'admin' ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'
                    }`}>
                      {u.role === 'admin' ? 'Администратор' : 'Аналитик'}
                    </span>
                  </td>
                  <td className="py-2">
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      u.is_active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
                    }`}>
                      {u.is_active ? 'Активен' : 'Заблокирован'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
