-- Pandoc Lua filter for the kirin paper style.
--   fenced divs with a box class      -> tcolorbox environments (title via attribute)
--   images / implicit figures         -> \includegraphics at \linewidth, no float
--   tables                            -> unbreakable tabular (pandoc emits page-breaking longtables)
--   table/figure + following caption  -> one minipage, so a page break never separates them
local boxes = {abstractbox=true, glancebox=true, resultbox=true, scopebox=true, auditbox=true, roadmapbox=true}

function Div(el)
  for _, c in ipairs(el.classes) do
    if boxes[c] then
      local title = el.attributes['title']
      local open = '\\begin{' .. c .. '}' .. (title and ('[{{' .. title .. '}}]') or '')
      return {pandoc.RawBlock('latex', open), table.unpack(el.content), pandoc.RawBlock('latex', '\\end{' .. c .. '}')}
    end
  end
end

local function graphic(src)
  return '\\par\\noindent\\includegraphics[width=\\linewidth]{' .. src .. '}\\par'
end

function Image(el)
  return pandoc.RawInline('latex', graphic(el.src))
end

function Figure(el)
  local raw = nil
  el:walk({RawInline = function(r) if r.text:find('includegraphics', 1, true) then raw = r.text end end,
           Image = function(img) raw = graphic(img.src) end})
  if raw then return pandoc.RawBlock('latex', raw) end
end

function Table(el)
  local tex = pandoc.write(pandoc.Pandoc({el}), 'latex')
  tex = tex:gsub('^%s*{\\def\\LTcaptype{none}[^\n]*\n', '')      -- pandoc's captionless-longtable group
  tex = tex:gsub('\n}%s*$', '\n')
  tex = tex:gsub('\\begin{longtable}%[%]', '\\begin{tabular}')
  tex = tex:gsub('\\bottomrule\\noalign{}\n\\endlastfoot\n', '')  -- footer rule moves to the end
  tex = tex:gsub('\\endfirsthead\n', ''):gsub('\\endhead\n', ''):gsub('\\endfoot\n', ''):gsub('\\endlastfoot\n', '')
  tex = tex:gsub('\\end{longtable}', '\\bottomrule\n\\end{tabular}')
  return pandoc.RawBlock('latex', '\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.10}\\vspace{5pt}\\noindent\n' .. tex)
end

local function caption_tex(b)
  if b.t == 'RawBlock' and b.text:find('kirinfigcaption', 1, true) then return b.text end
  if b.t == 'Para' and #b.content >= 1 and b.content[1].t == 'RawInline' and b.content[1].text:find('kirinfigcaption', 1, true) then
    return pandoc.write(pandoc.Pandoc({b}), 'latex')
  end
end

function Blocks(bs)
  local out = pandoc.List()
  local i = 1
  while i <= #bs do
    local b, nxt = bs[i], bs[i + 1]
    local body = nil
    if b.t == 'RawBlock' and (b.text:find('begin{tabular}', 1, true) or b.text:find('includegraphics', 1, true)) then body = b.text end
    local cap = body and nxt and caption_tex(nxt) or nil
    if cap then
      out:insert(pandoc.RawBlock('latex', '\\par\\noindent\\begin{minipage}{\\linewidth}\n' .. body .. '\n' .. cap .. '\n\\end{minipage}\\par\\vspace{4pt}'))
      i = i + 2
    else
      out:insert(b); i = i + 1
    end
  end
  return out
end
