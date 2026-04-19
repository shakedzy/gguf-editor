import { useState, useRef, ReactNode } from 'react'

interface Props {
  text: string
  children: ReactNode
  maxWidth?: number
}

export default function Tooltip({ text, children, maxWidth = 320 }: Props) {
  const [show, setShow] = useState(false)
  const [pos, setPos] = useState({ x: 0, y: 0 })
  const ref = useRef<HTMLSpanElement>(null)

  const handleEnter = () => {
    if (!ref.current) return
    const rect = ref.current.getBoundingClientRect()
    setPos({ x: rect.left, y: rect.bottom + 6 })
    setShow(true)
  }

  return (
    <>
      <span
        ref={ref}
        onMouseEnter={handleEnter}
        onMouseLeave={() => setShow(false)}
        className="inline-flex items-center cursor-help"
      >
        {children}
      </span>
      {show && (
        <div
          className="fixed z-[100] px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg shadow-xl text-xs text-gray-300 leading-relaxed normal-case tracking-normal"
          style={{
            left: Math.min(pos.x, window.innerWidth - maxWidth - 20),
            top: pos.y,
            maxWidth
          }}
        >
          {text}
        </div>
      )}
    </>
  )
}
