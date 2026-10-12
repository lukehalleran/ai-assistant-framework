import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import MarkdownMessage from './MarkdownMessage'

describe('MarkdownMessage math', () => {
  it('renders single-dollar amounts literally', () => {
    const text = 'The bill is $86.40 and a 15% tip is $12.96.'
    const { container } = render(<MarkdownMessage content={text} />)
    expect(container.textContent).toContain(text)
    expect(container.querySelector('.katex')).toBeNull()
  })
  it('still renders $$ display math', () => {
    const { container } = render(<MarkdownMessage content={'$$x^2$$'} />)
    expect(container.querySelector('.katex')).not.toBeNull()
  })
})
