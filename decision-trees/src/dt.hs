import System.IO
import Control.Monad
import System.Environment
import Data.List
import Data.Set (toList, fromList)
import Data.Function (on)

type DS = ([[Float]], [String], [String])


unique :: [String] -> [String]
unique = toList . fromList


nonEmpty :: [a] -> Bool
nonEmpty lst = not (null lst)


splitStr :: Char -> String -> [String]
splitStr delim str = filter nonEmpty (splitStr' delim str "")
    where
        splitStr' :: Char -> String -> String -> [String]
        splitStr' delim input current
            | null input          = [current]
            | head input == delim = current : splitStr' delim (tail input) ""
            | otherwise           = splitStr' delim (tail input) (current ++ [head input])


ticTacToeToFloat :: String -> Float
ticTacToeToFloat str
    | str == "x" = 0.0
    | str == "o" = 1.0
    | str == "b" = 2.0


floatToTicTacToe :: Float -> String
floatToTicTacToe num
    | num == 0.0 = "x"
    | num == 1.0 = "o"
    | num == 2.0 = "b"


readAttr :: String -> Float
readAttr = read


giniSteps = 20 :: Float


minimizeGini :: DS -> [Int] -> (Int, Float)
minimizeGini ds attributes = 
    let
        getGiniIndex = giniOnAttr ds
        bestSplits   = map getGiniIndex attributes
        zipped       = zip bestSplits attributes
        -- Empty structure
        best         = minimumBy (compare `on` (snd . fst)) zipped

        ((_, split), idx) = best
    in
        (idx, split)


giniOnAttr :: DS -> Int -> (Float, Float)
giniOnAttr ds attr = let
        (attrs, classes, classSet) = ds

        desAttr   = map (!! attr) attrs
        minVal    = minimum desAttr
        maxVal    = maximum desAttr
        step      = (maxVal - minVal) / giniSteps
        range     = [minVal-step,minVal..maxVal+step]
        giniIdx   = giniForSplit desAttr classes classSet
        allSplits = [(split, giniIdx split) | split <- range ]
        bestSplit = minimumBy (compare `on` snd) allSplits
    in
        bestSplit


giniForSplit :: [Float] -> [String] -> [String] -> Float -> Float
giniForSplit values classes classSet split =
    let 
        ziped    = zip values classes
        left     = filter ((<= split) . fst) ziped
        right    = filter ((>  split) . fst) ziped
        leftLen  = fromIntegral (length left)
        rightLen = fromIntegral (length right)
        total    = fromIntegral (length values)

        countClass :: [(Float, String)] -> String -> Int
        countClass lst cls = sum (map (\x -> if snd x == cls then 1 else 0) lst)

        leftCounts  = map (countClass left) classSet
        rightCounts = map (countClass right) classSet

        leftProbs  = map (((^2) . (/ leftLen)) . fromIntegral) leftCounts
        rightProbs = map (((^2) . (/ rightLen)) . fromIntegral) rightCounts

        giniLeft  = 1 - sum leftProbs
        giniRight = 1 - sum rightProbs

        giniTotal = giniLeft * (leftLen / total) + giniRight * (rightLen / total)
    in
        giniTotal


main = do
    args <- getArgs

    let infile    = head args
        depth     = read (args !! 1 ) :: Int
        header    = args !! 2 == "y"
        dataType  = args !! 3
        separator = head (args !! 4)

    putStrLn "Args Parsed"
    contents <- readFile infile

    let allLines      = lines contents
        untyped       = if header then tail allLines else allLines
        split         = map (splitStr separator) untyped
        attrCols      = map init split
        classes       = map last split
        uniqueClasses = unique classes
        attributes    = map (map (if dataType == "char" then ticTacToeToFloat else readAttr)) attrCols
        numAttributes = length (head attributes)
        attrIndices   = [0..(pred numAttributes)]
        df            = (attributes, classes, uniqueClasses)

    putStrLn "Dataset Loaded"

    print (minimizeGini df attrIndices)

    putStrLn "Shutting down"

