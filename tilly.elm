import Array exposing (Array, repeat, get, indexedMap, toList, slice)
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (..)
import Http
import Json.Decode as Decode

type Player =
    Me | Opponent | Nobody

board_size = 9


type alias Board = Array Player


type alias Model =
    {
        board: Board,
        values: TrainedValues,
        winner: Player
    }



type Msg = 
    Position (Int)
    | FetchedValues (Result Http.Error TrainedValues)


update_position : Int -> Player -> Int -> Player -> Player
update_position to_update substitute i current =
    if to_update == i && current == Nobody then substitute else current

is_position_free : Board -> Int -> Bool
is_position_free board position = 
    let
        current_value = getValueFromArrayWithDefault board Me position
    in
        current_value == Nobody



update : Msg -> Model -> (Model, Cmd Msg)
update msg {board, values, winner} =
    
    if haveWinnerOrDraw board then 
        ({board=board, values=values, winner=winner}, Cmd.none)
    else
    case msg of 
        Position position ->
            let
                newBoard= 
                    if is_position_free board position then
                        updateBoard board Me position
                        |>
                            aiMoveIfPossible Opponent values
                    else board
            in
            (
                {
                    board=newBoard,
                    values=values,
                    winner=(findWinner newBoard)
                },
                Cmd.none
            )
        FetchedValues (Ok newValues) ->
            ({board=board, values=newValues, winner=winner}, Cmd.none)
        _ -> ({board=board, values=values, winner=winner}, Cmd.none)



haveWinnerOrDraw :  Board -> Bool
haveWinnerOrDraw board =
    if (findWinner board) /= Nobody then True
    else haveDraw board


haveDraw : Board -> Bool
haveDraw board = 
    List.all (\x -> x /= Nobody) <| Array.toList board


aiMoveIfPossible : Player -> TrainedValues -> Board -> Board
aiMoveIfPossible currentPlayer values board =
    if haveWinnerOrDraw board then board
    else
        aiMove currentPlayer values board


type alias Line = List Int

lines : List Line
lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]]


getValueFromArrayWithDefault : (Array a) -> a -> Int -> a
getValueFromArrayWithDefault values default i =
    Maybe.withDefault default (get i values)

takeFromListWithDefault : (List a) -> a -> Int -> a
takeFromListWithDefault values default i = 
    if i == 0 then
        Maybe.withDefault default <| List.head values
    else
        takeFromListWithDefault (List.drop 1 values) default (i-1)


lineWinner : Board -> Line -> Player
lineWinner board line = 
    let 
        x = Debug.log "ok" "ok"
    in
    let
        a = getValueFromArrayWithDefault board Nobody 
            <| takeFromListWithDefault line 0 0 
        b = getValueFromArrayWithDefault board Nobody 
            <| takeFromListWithDefault line 0 1 
        c = getValueFromArrayWithDefault board Nobody 
            <| takeFromListWithDefault line 0 2 
        d = Debug.log "a, b, c" [a, b, c]
    in
        if (a == b) && (b == c) then a else Nobody


findWinner : Board -> Player
findWinner board = 
    List.map (lineWinner board) lines
    |> List.foldl (\player acc ->
        if acc /= Nobody then acc else player) Nobody


updateBoard: Board -> Player -> Int -> Board
updateBoard board current_player position =
    (indexedMap (update_position position current_player) board)


showPlayer : Player -> String
showPlayer player =
        case player of
                Me -> "X"
                Opponent -> "O"
                _ -> "-"


showPlayerButton : Int -> Int -> Player -> Html Msg
showPlayerButton offset position player = button [ onClick (Position (position+offset))] [text (showPlayer player)]


showPlayerRow : Int-> Array Player -> Html Msg
showPlayerRow offset row = div [class "ticrow"] <| List.indexedMap (showPlayerButton offset) (toList row)


showWinner : Board -> Player -> Html Msg
showWinner board winner = 
    if haveDraw board then
        div [class "result" ] [text "Draw"]
    else
    case winner of 
        Nobody -> div [class "result"] [text "Tic Tac Toe"]
        Me -> div [class "result" ] [text "You Win"]
        Opponent -> div [class "result" ] [text "I Win"]


view : Model -> Html Msg
view model =
        div [] [
                showWinner model.board model.winner,
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


-- defaultValues : TrainedValues
-- defaultValues =
--     repeat (3^9) 0.0


--- get default values


getTrainedValues : Cmd Msg
getTrainedValues =
  let
    url =
      "/values.json?"
  in
    Http.send FetchedValues (Http.get url decodeInitialValues)


decodeInitialValues : Decode.Decoder TrainedValues
decodeInitialValues =
    Decode.at [] <| Decode.array Decode.float


init : (Model, Cmd Msg)
init = (
    {board=repeat board_size Nobody, 
        values=repeat (3^9) 0.0,
        winner=Nobody},
    getTrainedValues
    )


-- SUBSCRIPTIONS


subscriptions : Model -> Sub Msg
subscriptions model =
  Sub.none


main =
  Html.program
    { init = init
    , view = view
    , update = update
    , subscriptions = subscriptions
    }
