-- Pandoc Lua filter: the kirin-structured markdown -> plain TMLR LaTeX.
--   kirin boxes            -> \paragraph{Title.} + text (abstract box -> abstract environment;
--                             the roadmap box becomes an "Organization" paragraph at the end of Section 1)
--   "# 1. Title" headings  -> unnumbered text, LaTeX numbers the sections
--   \kirinpart{Appendix X}{Title} -> \appendix (once) + \section{Title}
--   tables                 -> table floats with the caption above (pandoc longtable -> tabular)
--   images + caption       -> figure floats with the caption below
--   ::: {#refs}            -> \bibliographystyle{tmlr}\bibliography{references}
local plain_boxes = {glancebox=true, resultbox=true, scopebox=true, auditbox=true}
local roadmap = nil
local appendix_started = false
local ntab, nfig = 0, 0

local function box_title(t)
  if not t then return nil end
  t = t:gsub('%s*%-%-%-%s*', ': ')          -- "Result 1 --- claim" -> "Result 1: claim"
  if not t:match('[%.%?!]$') then t = t .. '.' end
  return t
end

function Div(el)
  for _, c in ipairs(el.classes) do
    if c == 'abstractbox' then
      return {pandoc.RawBlock('latex', '\\begin{abstract}'), table.unpack(el.content), pandoc.RawBlock('latex', '\\end{abstract}')}
    elseif c == 'roadmapbox' then
      roadmap = el.content; return {}
    elseif plain_boxes[c] then
      local t = box_title(el.attributes['title'])
      local out = pandoc.List()
      if t then out:insert(pandoc.RawBlock('latex', '\\paragraph{' .. t .. '}')) end
      for _, b in ipairs(el.content) do out:insert(b) end
      return out
    end
  end
  if el.identifier == 'refs' then
    return pandoc.RawBlock('latex', '\\bibliographystyle{tmlr}\n\\bibliography{references}')
  end
end

function Header(el)
  -- strip the manual "1." / "2.1" prefixes
  if #el.content > 0 and el.content[1].t == 'Str' and el.content[1].text:match('^%d+%.?%d*%.?$') then
    el.content:remove(1)
    if #el.content > 0 and el.content[1].t == 'Space' then el.content:remove(1) end
  end
  return el
end

local function graphic(src) return '\\includegraphics[width=\\linewidth]{' .. src .. '}' end
function Image(el) return pandoc.RawInline('latex', graphic(el.src)) end
function Figure(el)
  local raw = nil
  el:walk({RawInline = function(r) if r.text:find('includegraphics', 1, true) then raw = r.text end end,
           Image = function(img) raw = graphic(img.src) end})
  if raw then return pandoc.RawBlock('latex', raw) end
end

function Table(el)
  local tex = pandoc.write(pandoc.Pandoc({el}), 'latex')
  tex = tex:gsub('^%s*{\\def\\LTcaptype{none}[^\n]*\n', ''):gsub('\n}%s*$', '\n')
  tex = tex:gsub('\\begin{longtable}%[%]', '\\begin{tabular}')
  tex = tex:gsub('\\bottomrule\\noalign{}\n\\endlastfoot\n', '')
  tex = tex:gsub('\\endfirsthead\n', ''):gsub('\\endhead\n', ''):gsub('\\endfoot\n', ''):gsub('\\endlastfoot\n', '')
  tex = tex:gsub('\\end{longtable}', '\\bottomrule\n\\end{tabular}')
  return pandoc.RawBlock('latex', tex)
end

local function kirinpart(text)
  local label, title = text:match('\\kirinpart{(.-)}{(.-)}')
  if not label then return nil end
  local pre = ''
  if not appendix_started then appendix_started = true; pre = '\\appendix\n' end
  return pandoc.RawBlock('latex', pre .. '\\section{' .. title .. '}\\label{app:' .. label:gsub('Appendix ', '') .. '}')
end

local function caption_text(b)
  local raw = nil
  if b.t == 'RawBlock' and b.text:find('kirinfigcaption', 1, true) then raw = b.text
  elseif b.t == 'Para' and #b.content >= 1 and b.content[1].t == 'RawInline' and b.content[1].text:find('kirinfigcaption', 1, true) then
    raw = pandoc.write(pandoc.Pandoc({b}), 'latex') end
  if not raw then return nil end
  -- \kirinfigcaption{Table 1.}{text}  -> text (balanced-brace scan for the second argument)
  local i = raw:find('\\kirinfigcaption', 1, true); local p = raw:find('{', i, true); local depth = 0; local j = p
  local args = {}
  for k = 1, 2 do
    depth = 0; local s = j
    while j <= #raw do
      local ch = raw:sub(j, j)
      if ch == '{' then depth = depth + 1 elseif ch == '}' then depth = depth - 1; if depth == 0 then break end end
      j = j + 1
    end
    args[k] = raw:sub(s + 1, j - 1); j = raw:find('{', j, true)
    if not j then break end
  end
  return args[2]
end

function Blocks(bs)
  local out = pandoc.List()
  local i = 1
  while i <= #bs do
    local b, nxt = bs[i], bs[i + 1]
    -- appendix parts
    if b.t == 'RawBlock' and b.text:find('\\kirinpart', 1, true) then out:insert(kirinpart(b.text)); i = i + 1
    elseif b.t == 'Para' and #b.content == 1 and b.content[1].t == 'RawInline' and b.content[1].text:find('\\kirinpart', 1, true) then
      out:insert(kirinpart(b.content[1].text)); i = i + 1
    else
      local cap = nxt and caption_text(nxt) or nil
      if cap and b.t == 'RawBlock' and b.text:find('begin{tabular}', 1, true) then
        ntab = ntab + 1
        local setup = '\\centering\\small\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{1.08}\n'
        if appendix_started then
          -- appendix tables are set in place (an unbreakable block right after their heading), not floated
          out:insert(pandoc.RawBlock('latex', '\\noindent\\begin{minipage}{\\linewidth}\n\\captionof{table}{' .. cap .. '}\\label{tab:' .. ntab .. '}\n' .. setup .. b.text .. '\\end{minipage}\\par\\vspace{\\baselineskip}'))
        else
          out:insert(pandoc.RawBlock('latex', '\\begin{table}[htbp]\n\\caption{' .. cap .. '}\\label{tab:' .. ntab .. '}\n' .. setup .. b.text .. '\\end{table}'))
        end
        i = i + 2
      elseif cap and b.t == 'RawBlock' and b.text:find('includegraphics', 1, true) then
        nfig = nfig + 1
        out:insert(pandoc.RawBlock('latex', '\\begin{figure}[htbp]\n\\centering\n' .. b.text .. '\n\\caption{' .. cap .. '}\\label{fig:' .. nfig .. '}\n\\end{figure}'))
        i = i + 2
      else out:insert(b); i = i + 1 end
    end
  end
  return out
end

local BROADER = [[
\subsubsection*{Broader Impact Statement}
This paper evaluates a readout method on one frozen open-weights model. Hidden-state readouts could complement observation of visible reasoning, but only after establishing what their scores measure. No monitoring system, deception detector, or chain-of-thought baseline is evaluated here, and the study does not establish performance on arbitrary concepts, other models, or monitoring under adversarial pressure.]]

function Pandoc(doc)
  -- TMLR order: body, broader impact statement, references, appendices
  local blocks = doc.blocks
  local bib_idx, app_idx = nil, nil
  for idx, b in ipairs(blocks) do
    if b.t == 'RawBlock' and b.text:find('\\bibliographystyle', 1, true) then bib_idx = idx end
    if b.t == 'RawBlock' and b.text:find('^\\appendix') and not app_idx then app_idx = idx end
  end
  if bib_idx and app_idx and bib_idx > app_idx then
    local bib = blocks:remove(bib_idx)
    -- drop the now-empty "References" heading that preceded the refs div
    for k = #blocks, 1, -1 do
      if blocks[k].t == 'Header' and pandoc.utils.stringify(blocks[k]) == 'References' then blocks:remove(k) break end
    end
    blocks:insert(app_idx, bib)
    blocks:insert(app_idx, pandoc.RawBlock('latex', BROADER))
  end
  -- place the roadmap text as an "Organization" paragraph just before the second top-level section
  if roadmap then
    local blocks = doc.blocks; local seen = 0
    for idx, b in ipairs(blocks) do
      if b.t == 'Header' and b.level == 1 then
        seen = seen + 1
        if seen == 2 then
          local ins = pandoc.List({pandoc.RawBlock('latex', '\\paragraph{Organization.}')})
          for _, rb in ipairs(roadmap) do ins:insert(rb) end
          for k = #ins, 1, -1 do blocks:insert(idx, ins[k]) end
          break
        end
      end
    end
  end
  return doc
end
