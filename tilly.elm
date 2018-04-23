import Array exposing (Array, repeat, get, indexedMap, toList, slice)
import Html exposing (Html, button, div, text)
import Html.Events exposing (onClick)

type Player =
    Me | Opponent | Nobody

board_size = 9

type alias Model =
    {
        board: Array Player,
        current_player: Player
    }

model : Model
model = Model (repeat board_size Nobody) Me

type alias Msg = {position: Int}  -- int: position; player: player


update_position : Int -> Player -> Int -> Player -> Player
update_position to_update substitute i current =
    if to_update == i then substitute else current

update : Msg -> Model -> Model
update {position} {board, current_player} =
    Model (indexedMap (update_position position current_player) board) (if current_player == Me then Opponent else Me)


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



main =
  Html.beginnerProgram
    { model = model
    , view = view
    , update = update
    }
