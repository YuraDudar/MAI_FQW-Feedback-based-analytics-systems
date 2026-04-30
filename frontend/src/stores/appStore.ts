import { create } from 'zustand'
import type { Product } from '../types'

interface AppState {
  products: Product[]
  selectedProductId: number | null
  setProducts: (products: Product[]) => void
  setSelectedProduct: (id: number | null) => void
  addProduct: (product: Product) => void
  removeProduct: (id: number) => void
}

export const useAppStore = create<AppState>()(set => ({
  products: [],
  selectedProductId: null,
  setProducts: products => set({ products }),
  setSelectedProduct: id => set({ selectedProductId: id }),
  addProduct: product => set(state => ({ products: [product, ...state.products] })),
  removeProduct: id => set(state => ({ products: state.products.filter(p => p.product_id !== id) })),
}))
