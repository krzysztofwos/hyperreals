import Hyperreals.PeriodicCore
import Lean.Data.Json

/-!
JSON-lines transport for the verified periodic kernel. Parsing and native code
execution are separate implementation boundaries. The comparison and transition
functions called here are the functions whose specifications are proved in Lean.
-/

open Lean Hyperreals.Periodic

private def parseSupport (json : Json) : Except String Support := do
  let values ← json.getArr?
  unless values.size = 2 do throw "support must contain exactly two Booleans"
  return ⟨← values[0]!.getBool?, ← values[1]!.getBool?⟩

private partial def parseExpr (json : Json) : Except String Hyperreals.Periodic.Expr := do
  let values ← json.getArr?
  let tag ← (← json.getArrVal? 0).getStr?
  if tag = "const" then
    unless values.size = 3 do throw "const expects numerator and denominator strings"
    let numeratorText ← values[1]!.getStr?
    let denominatorText ← values[2]!.getStr?
    let some numerator := numeratorText.toInt? | throw "invalid numerator integer"
    let some denominator := denominatorText.toNat? | throw "invalid denominator integer"
    if denominator = 0 then throw "denominator must be positive"
    return .constant (mkRat numerator denominator)
  else if tag = "alt" then
    unless values.size = 1 do throw "alt expects no arguments"
    return .alternating
  else
    unless values.size = 3 do throw "binary expression expects two arguments"
    let left ← parseExpr values[1]!
    let right ← parseExpr values[2]!
    match tag with
    | "add" => return .add left right
    | "sub" => return .sub left right
    | "mul" => return .mul left right
    | _ => throw s!"unsupported expression operator: {tag}"

private def supportJson (support : Support) : Json :=
  Json.arr #[Json.bool support.even, Json.bool support.odd]

private def handleRequest (request : Json) : Except String Json := do
  let support ← parseSupport (← request.getObjVal? "support")
  unless support.nonempty do throw "input support must be nonempty"
  let operator ← (← request.getObjVal? "op").getStr?
  let comparison ← match operator with
    | "lt" => pure Comparison.lt
    | "eq" => pure Comparison.eq
    | _ => throw "op must be lt or eq"
  let left ← parseExpr (← request.getObjVal? "left")
  let right ← parseExpr (← request.getObjVal? "right")
  let predicate := comparison.compile left right
  let positive := support.inter predicate
  let negative := support.inter predicate.compl
  let mut fields := [
    ("predicate", supportJson predicate),
    ("trueSupport", supportJson positive),
    ("falseSupport", supportJson negative),
    ("canBeTrue", Json.bool positive.nonempty),
    ("canBeFalse", Json.bool negative.nonempty)]
  match request.getObjVal? "choice" with
  | .error _ =>
    fields := fields ++ [("accepted", Json.null), ("support", Json.null)]
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
