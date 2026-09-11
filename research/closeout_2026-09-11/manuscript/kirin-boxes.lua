-- Pandoc Lua filter: fenced divs with a kirin box class become tcolorbox environments.
-- ::: {.resultbox title="Result 1 --- ..."}  ->  \begin{resultbox}[{Result 1 --- ...}] ... \end{resultbox}
local boxes = {abstractbox=true, glancebox=true, resultbox=true, scopebox=true, auditbox=true, roadmapbox=true}
function Div(el)
  for _, c in ipairs(el.classes) do
    if boxes[c] then
      local title = el.attributes['title']
      local open = '\\begin{' .. c .. '}' .. (title and ('[{{' .. title .. '}}]') or '')
      return {pandoc.RawBlock('latex', open), table.unpack(el.content), pandoc.RawBlock('latex', '\\end{' .. c .. '}')} -- content kept
    end
  end
end
-- Images: no float, full width, no pandoc caption (the \kirinfigcaption line in the source carries the caption).
function Image(el)
  return pandoc.RawInline('latex', '\\par\\noindent\\includegraphics[width=\\linewidth]{' .. el.src .. '}\\par')
end

-- Pandoc 3 wraps a lone image in a Figure block with a default caption. The Image handler above has already
-- rewritten the image into a RawInline; unwrap the Figure so nothing floats and no default caption is added.
function Figure(el)
  local raw = nil
  el:walk({RawInline = function(r) if r.text:find('includegraphics', 1, true) then raw = r.text end end,
           Image = function(img) raw = '\\par\\noindent\\includegraphics[width=\\linewidth]{' .. img.src .. '}\\par' end})
  if raw then return pandoc.RawBlock('latex', raw) end
end
