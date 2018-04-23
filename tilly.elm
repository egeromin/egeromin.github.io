import Array exposing (Array, repeat, get, indexedMap, toList, slice)
import Html exposing (Html, button, div, text)
import Html.Events exposing (onClick)

type Player =
    Me | Opponent | Nobody

board_size = 9


type alias Board = Array Player


type alias Model =
    {
        board: Board
    }

model : Model
model = Model (repeat board_size Nobody)

type alias Msg = {position: Int}  -- int: position; player: player


update_position : Int -> Player -> Int -> Player -> Player
update_position to_update substitute i current =
    if to_update == i then substitute else current

update : Msg -> Model -> Model
update {position} {board} =
    Model (
        updateBoard board Me position
        |>
            aiMove Opponent defaultValues
    )

updateBoard: Board -> Player -> Int -> Board
updateBoard board current_player position =
    (indexedMap (update_position position current_player) board)


showPlayer : Player -> String
showPlayer player =
        case player of
                Me -> "X"
                Opponent -> "O"
                _ -> "E"


showPlayerButton : Int -> Int -> Player -> Html Msg
showPlayerButton offset position player = button [ onClick {position=position+offset}] [text (showPlayer player)]


showPlayerRow : Int-> Array Player -> Html Msg
showPlayerRow offset row = div [] <| List.indexedMap (showPlayerButton offset) (toList row)


view : Model -> Html Msg
view model =
        div [] [
                showPlayerRow 0 (slice 0 3 model.board),
                showPlayerRow 3 (slice 3 6 model.board),
                showPlayerRow 6 (slice 6 9 model.board)
                ]



emptyIndices: Board -> EmptyIndices
emptyIndices board =
    board
        |> Array.indexedMap (,)
        |> toList
        |> List.filterMap (\(idx,player) -> if player == Nobody then Just idx else Nothing)

type alias EmptyIndices = List Int
type alias NextStates = List Board

filledBoards : Board -> Player -> EmptyIndices -> NextStates
filledBoards board currentPlayer emptyIndices =
    List.map (updateBoard board currentPlayer) emptyIndices


nextStates : Board -> Player -> List Board
nextStates board currentPlayer =
    emptyIndices board
        |> filledBoards board currentPlayer

indexFromBoard: Board -> Int
indexFromBoard board =
    board
        |> Array.map intFromPlayer
        |> Array.indexedMap (,)
        |> Array.foldl (\(idx, int) acc ->
                acc + int * 3 ^ idx
            ) 0


intToTernary: Int -> List Int
intToTernary int =
    let
        remainder = int % 3
    in
    if int == 0 then [] else remainder::(intToTernary ((int-remainder) // 3))


padList: Int -> x -> List x -> List x
padList len default list =
    List.concat [list, (List.repeat (len - List.length list) default)]



boardFromIndex : Int -> Board
boardFromIndex i =
    intToTernary i
        |> padList 9 0
        |> List.map playerFromInt
        |> Array.fromList


intFromPlayer : Player -> Int
intFromPlayer player =
    case player of
        Me -> 2
        Opponent -> 1
        Nobody -> 0

playerFromInt : Int -> Player
playerFromInt int =
    case int of
        2 -> Me
        1 -> Opponent
        _ -> Nobody

type alias TrainedValues = Array Float

getValueFromArray: TrainedValues -> Int -> Float
getValueFromArray values i =
    Maybe.withDefault  -1 (get i values)

aiMove : Player -> TrainedValues -> Board -> Board
aiMove currentPlayer values board =
    let
        a = nextStates board currentPlayer
        |> List.map indexFromBoard
        |> List.foldl (\i acc ->
                let
                    b = Debug.log " getValueFromArray values i" ( getValueFromArray values i  > getValueFromArray values acc )
                in
                if getValueFromArray values i > getValueFromArray values acc then
                    i
                else
                    acc
            ) -1
        d = Debug.log "a" a
    in
    a |>  boardFromIndex


defaultValues : TrainedValues
defaultValues =
    repeat (3^9) 0.0

main =
  Html.beginnerProgram
    { model = model
    , view = view
    , update = update
    }
