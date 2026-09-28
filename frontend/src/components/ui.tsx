import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from 'react'
import type { MetadataStatus } from '../types'

export function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(' ')
}

export function Card({
  title,
  eyebrow,
  actions,
  children,
  className,
}: {
  title?: ReactNode
  eyebrow?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={cx('rounded-xl border border-line bg-surface shadow-[0_1px_2px_rgba(15,23,42,0.04)]', className)}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-3 border-b border-line px-5 py-3.5">
          <div>
            {eyebrow && <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{eyebrow}</div>}
            {title && <h3 className="text-[15px] font-semibold text-ink">{title}</h3>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

export function Button({
  variant = 'secondary',
  size = 'md',
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: 'sm' | 'md' | 'lg' }) {
  const styles: Record<ButtonVariant, string> = {
    primary: 'bg-brand text-brand-ink hover:brightness-110 shadow-sm',
    secondary: 'border border-line-strong bg-surface text-ink hover:bg-surface-2',
    ghost: 'text-ink-2 hover:bg-surface-2 hover:text-ink',
    danger: 'border border-line-strong bg-surface text-critical-text hover:bg-critical-soft',
  }
  const sizes = { sm: 'h-8 px-3 text-[13px]', md: 'h-9 px-3.5 text-sm', lg: 'h-11 px-5 text-[15px]' }
  return (
    <button
      type="button"
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-lg font-medium transition disabled:cursor-not-allowed disabled:opacity-50',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand',
        styles[variant],
        sizes[size],
        className,
      )}
      {...props}
    />
  )
}

export function Field({
  label,
  hint,
  children,
  className,
  auto,
}: {
  label: ReactNode
  hint?: ReactNode
  children: ReactNode
  className?: string
  auto?: boolean
}) {
  return (
    <label className={cx('block', className)}>
      <span className="mb-1.5 flex items-center justify-between text-[13px] font-medium text-ink-2">
        <span>{label}</span>
        {auto && <span className="text-[11px] font-normal text-ink-3">Auto</span>}
      </span>
      {children}
      {hint && <span className="mt-1 block text-xs text-ink-3">{hint}</span>}
    </label>
  )
}

const inputBase =
  'h-9 w-full rounded-lg border border-line-strong bg-surface px-3 text-sm text-ink placeholder:text-ink-3 focus:border-brand focus:outline-none focus:ring-2 focus:ring-brand/20 disabled:bg-surface-2 disabled:text-ink-3'

export function Input(props: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={cx(inputBase, props.className)} />
}

export function NumberInput({
  value,
  onChange,
  ...props
}: Omit<InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'type'> & {
  value: number | null | undefined
  onChange: (v: number | null) => void
}) {
  return (
    <input
      type="number"
      inputMode="decimal"
      value={value ?? ''}
      onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
      {...props}
      className={cx(inputBase, 'tabular', props.className)}
    />
  )
}

export function Select({ children, ...props }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select {...props} className={cx(inputBase, 'pr-8', props.className)}>
      {children}
    </select>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean
  onChange: (v: boolean) => void
  label: ReactNode
  hint?: ReactNode
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 py-1">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cx(
          'relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand',
          checked ? 'bg-brand' : 'bg-line-strong',
        )}
      >
        <span
          className={cx(
            'absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow transition',
            checked && 'translate-x-4',
          )}
        />
      </button>
      <span className="text-sm">
        <span className="text-ink">{label}</span>
        {hint && <span className="block text-xs text-ink-3">{hint}</span>}
      </span>
    </label>
  )
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  return (
    <div className="inline-flex rounded-lg border border-line-strong bg-surface-2 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={cx(
            'h-7 rounded-md px-3 text-[13px] font-medium transition',
            value === o.value ? 'bg-surface text-ink shadow-sm' : 'text-ink-2 hover:text-ink',
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Tabs<T extends string>({
  value,
  tabs,
  onChange,
}: {
  value: T
  tabs: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  return (
    <div className="flex gap-1 overflow-x-auto border-b border-line" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.value}
          role="tab"
          aria-selected={value === t.value}
          type="button"
          onClick={() => onChange(t.value)}
          className={cx(
            '-mb-px border-b-2 px-3 py-2.5 text-sm font-medium whitespace-nowrap transition',
            value === t.value ? 'border-brand text-brand-text' : 'border-transparent text-ink-2 hover:text-ink',
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

type Tone = 'neutral' | 'brand' | 'good' | 'warning' | 'critical'

export function Badge({ tone = 'neutral', children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  const tones: Record<Tone, string> = {
    neutral: 'bg-info-soft text-ink-2 border-line',
    brand: 'bg-brand-soft text-brand-text border-transparent',
    good: 'bg-good-soft text-good-text border-transparent',
    warning: 'bg-warning-soft text-warning-text border-transparent',
    critical: 'bg-critical-soft text-critical-text border-transparent',
  }
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px] font-semibold tracking-wide uppercase',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

export function StatusBadge({ status }: { status: MetadataStatus }) {
  const map: Record<MetadataStatus, { tone: Tone; label: string; icon: string }> = {
    verified: { tone: 'good', label: 'Verified', icon: '✓' },
    imported: { tone: 'brand', label: 'Imported', icon: '↓' },
    user_defined: { tone: 'warning', label: 'User-defined', icon: '✎' },
  }
  const m = map[status]
  return (
    <Badge tone={m.tone}>
      <span aria-hidden>{m.icon}</span>
      {m.label}
    </Badge>
  )
}

export function ConfidenceBadge({ label }: { label: string }) {
  const bench = label.toUpperCase().startsWith('BENCH')
  return (
    <Badge tone={bench ? 'good' : 'neutral'}>
      <span aria-hidden>{bench ? '◉' : '≈'}</span>
      {bench ? 'Benchmark-based' : 'Estimated'}
    </Badge>
  )
}

export function Stat({ label, value, sub }: { label: ReactNode; value: ReactNode; sub?: ReactNode }) {
  return (
    <div>
      <div className="text-xs text-ink-3">{label}</div>
      <div className="text-[15px] font-semibold text-ink">{value}</div>
      {sub && <div className="text-xs text-ink-3">{sub}</div>}
    </div>
  )
}

export function KV({ rows }: { rows: [ReactNode, ReactNode][] }) {
  return (
    <dl className="divide-y divide-line text-sm">
      {rows.map(([k, v], i) => (
        <div key={i} className="flex items-baseline justify-between gap-4 py-1.5">
          <dt className="text-ink-2">{k}</dt>
          <dd className="tabular text-right font-medium text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  )
}

export function Alert({ tone, children }: { tone: 'info' | 'warning' | 'critical' | 'good'; children: ReactNode }) {
  const styles = {
    info: 'bg-info-soft text-ink-2',
    good: 'bg-good-soft text-good-text',
    warning: 'bg-warning-soft text-warning-text',
    critical: 'bg-critical-soft text-critical-text',
  }
  const icons = { info: 'ℹ', good: '✓', warning: '⚠', critical: '⛔' }
  return (
    <div className={cx('flex gap-2.5 rounded-lg px-3 py-2 text-sm', styles[tone])} role={tone === 'critical' ? 'alert' : undefined}>
      <span aria-hidden className="mt-px shrink-0">
        {icons[tone]}
      </span>
      <div>{children}</div>
    </div>
  )
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-label="Loading"
      className={cx('inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent', className)}
    />
  )
}

export function PageHeader({ title, description, actions }: { title: string; description?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink">{title}</h1>
        {description && <p className="mt-1 max-w-3xl text-sm text-ink-2">{description}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  )
}

export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line bg-surface">
      <table className="tabular w-full text-left text-sm [&_td]:px-3 [&_td]:py-2 [&_th]:px-3 [&_th]:py-2 [&_th]:text-xs [&_th]:font-semibold [&_th]:text-ink-3 [&_thead]:bg-surface-2 [&_tr]:border-b [&_tr]:border-line [&_tbody_tr:last-child]:border-0">
        {children}
      </table>
    </div>
  )
}

/** Small "?" that reveals help text on hover or keyboard focus. */
export function InfoTip({ children, label = 'More information' }: { children: ReactNode; label?: string }) {
  return (
    <span className="group relative inline-flex align-middle">
      <button
        type="button"
        aria-label={label}
        className="flex h-4 w-4 items-center justify-center rounded-full border border-line-strong text-[10px] font-semibold text-ink-3 hover:border-ink-3 hover:text-ink focus-visible:outline-2 focus-visible:outline-brand"
      >
        ?
      </button>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute bottom-full left-1/2 z-30 mb-2 w-64 -translate-x-1/2 rounded-lg border border-line bg-surface p-2.5 text-xs leading-relaxed font-normal text-ink-2 opacity-0 shadow-lg transition group-focus-within:visible group-focus-within:opacity-100 group-hover:visible group-hover:opacity-100"
      >
        {children}
      </span>
    </span>
  )
}

/** Numbered step heading used by the calculator form. */
export function StepHeader({ step, title, hint, aside }: { step: number; title: string; hint?: ReactNode; aside?: ReactNode }) {
  return (
    <div className="mb-3 flex items-start justify-between gap-3">
      <div className="flex items-start gap-2.5">
        <span className="tabular mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-brand text-[11px] font-semibold text-brand-ink">
          {step}
        </span>
        <div>
          <h2 className="text-[15px] leading-5 font-semibold text-ink">{title}</h2>
          {hint && <p className="mt-0.5 text-xs text-ink-3">{hint}</p>}
        </div>
      </div>
      {aside}
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-xl border border-dashed border-line-strong p-8 text-center text-sm text-ink-3">{children}</div>
}
