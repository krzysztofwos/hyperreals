import Hyperreals.ResidueRuntimeCore
import Lean.Data.Json

/-! JSON transport for the proved arbitrary-period Laurent computation functions.
Parsing and native execution remain explicit, tested trust boundaries. -/

open Lean Hyperreals.Residue

private def parseSupport (json : Json) : Except String Support := do
  let values ← json.getArr?
  if values.isEmpty then throw "support must have positive period"
  return (← values.mapM Json.getBool?).toList

private def parseRat (numeratorJson denominatorJson : Json) : Except String Rat := do
  let some numerator := (← numeratorJson.getStr?).toInt? | throw "invalid numerator integer"
  let some denominator := (← denominatorJson.getStr?).toNat? | throw "invalid denominator integer"
  if denominator = 0 then throw "denominator must be positive"
  return mkRat numerator denominator

private partial def parseExpr (json : Json) : Except String Expr := do
  let values ← json.getArr?
  let tag ← (← json.getArrVal? 0).getStr?
  match tag with
  | "const" =>
    unless values.size = 3 do throw "const expects numerator and denominator strings"
    return .constant (← parseRat values[1]! values[2]!)
  | "periodic" =>
    unless values.size = 2 do throw "periodic expects one nonempty rational table"
    let table ← values[1]!.getArr?
    if table.isEmpty then throw "periodic table must be nonempty"
    let entries ← table.mapM fun entry => do
      let parts ← entry.getArr?
      unless parts.size = 2 do throw "periodic entries must be rational pairs"
      parseRat parts[0]! parts[1]!
    return .periodic entries.toList
  | "alt" | "index" | "invn" =>
    unless values.size = 1 do throw "primitive expects no arguments"
    return if tag = "alt" then .periodic [1, -1] else if tag = "index" then .index
      else .reciprocalIndex
  | "divMonomial" =>
    unless values.size = 5 do throw "divMonomial expects expression, numerator, denominator, power"
    let argument ← parseExpr values[1]!
    let coefficient ← parseRat values[2]! values[3]!
    let some power := (← values[4]!.getStr?).toInt? | throw "invalid monomial power"
    if coefficient = 0 then throw "monomial divisor must be nonzero"
    return .divMonomial argument coefficient power
  | "add" | "sub" | "mul" =>
    unless values.size = 3 do throw "binary expression expects two arguments"
    let left ← parseExpr values[1]!
    let right ← parseExpr values[2]!
    return if tag = "add" then .add left right else if tag = "sub" then .sub left right
      else .mul left right
  | _ => throw s!"unsupported expression operator: {tag}"

private def supportJson (support : Support) : Json :=
  Json.arr (support.map Json.bool).toArray

private def rationalJson (value : Rat) : Json :=
  Json.arr #[Json.str (toString value.num), Json.str (toString value.den)]

private def handleRequest (request : Json) : Except String Json := do
  let support ← parseSupport (← request.getObjVal? "support")
  unless support.nonempty do throw "input support must be nonempty"
  let operator ← (← request.getObjVal? "op").getStr?
  let left ← parseExpr (← request.getObjVal? "left")
  unless left.valid do throw "invalid expression"
  if operator = "standardPart" then
    let value := match standardPart support left with
      | some value => rationalJson value
      | none => Json.null
    return Json.mkObj [("value", value), ("support", supportJson support)]
  let comparison ← match operator with
    | "lt" => pure Hyperreals.Periodic.Comparison.lt
    | "eq" => pure Hyperreals.Periodic.Comparison.eq
    | _ => throw "op must be lt, eq, or standardPart"
  let right ← parseExpr (← request.getObjVal? "right")
  unless right.valid do throw "invalid expression"
  let compiled := compile comparison left right
  let predicate := compiled.mask
  let positive := support.inter predicate
  let negative := support.inter predicate.compl
  let mut fields := [
    ("predicate", supportJson predicate),
    ("cutoff", Json.str (toString compiled.cutoff)),
    ("trueSupport", supportJson positive),
    ("falseSupport", supportJson negative),
    ("canBeTrue", Json.bool positive.nonempty),
    ("canBeFalse", Json.bool negative.nonempty)]
  match request.getObjVal? "choice" with
  | .error _ => fields := fields ++ [("accepted", Json.null), ("support", Json.null)]
  | .ok value =>
    let choice ← value.getBool?
    match commit support ⟨comparison, left, right, choice⟩ with
    | none => fields := fields ++ [("accepted", Json.bool false), ("support", Json.null)]
    | some next =>
      fields := fields ++ [("accepted", Json.bool true), ("support", supportJson next)]
  return Json.mkObj fields

def main : IO Unit := do
  let input ← IO.getStdin
  let output ← IO.getStdout
  repeat
    let line ← input.getLine
    if line.isEmpty then break
    let result := Json.parse line >>= handleRequest
    let response := match result with
      | .ok value => value
      | .error message => Json.mkObj [("error", Json.str message)]
    output.putStrLn response.compress
    output.flush
