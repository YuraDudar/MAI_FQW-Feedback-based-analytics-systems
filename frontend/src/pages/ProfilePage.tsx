import { useAuthStore } from '../stores/authStore'
import { User, Shield, Calendar } from 'lucide-react'

export default function ProfilePage() {
  const { user } = useAuthStore()

  if (!user) return null

  const fields = [
    { label: 'Имя пользователя', value: user.username, icon: User },
    { label: 'Email', value: user.email, icon: User },
    { label: 'Роль', value: user.role === 'admin' ? 'Администратор' : 'Аналитик', icon: Shield },
    { label: 'Дата регистрации', value: new Date(user.created_at).toLocaleDateString('ru-RU'), icon: Calendar },
  ]

  return (
    <div className="max-w-lg space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Личный кабинет</h1>
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        <div className="flex items-center gap-4 mb-6 pb-6 border-b border-gray-100">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
            <span className="text-2xl font-bold text-blue-600">
              {user.username[0].toUpperCase()}
            </span>
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900">{user.username}</h2>
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              user.role === 'admin' ? 'bg-red-100 text-red-700' : 'bg-blue-100 text-blue-700'
            }`}>
              {user.role === 'admin' ? 'Администратор' : 'Аналитик'}
            </span>
          </div>
        </div>
        <div className="space-y-4">
          {fields.map(({ label, value, icon: Icon }) => (
            <div key={label} className="flex items-center gap-3">
              <Icon className="w-4 h-4 text-gray-400 shrink-0" />
              <div>
                <p className="text-xs text-gray-500">{label}</p>
                <p className="text-sm font-medium text-gray-900">{value}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
