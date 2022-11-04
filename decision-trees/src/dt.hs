import System.IO
import Control.Monad
import System.Environment
import Data.List
import Data.Set (toList, fromList)
import Data.Function (on)
import Control.Arrow (ArrowChoice(left))

data DS = DS [[Float]] [String] [String]

data Tree = Root Int Float Tree Tree | Leaf String

filterByMask :: [a] -> [Bool] -> [a]
filterByMask lst mask = [value | (value, flag) <- zip lst mask, flag]

mode :: Ord a => [a] -> a
mode lst = maxVal
    where
        values = unique lst
        countValue desired = (sum [1 :: Int | val <- lst, val == desired], desired)
        counts = map countValue values
        (count, maxVal) = maximumBy (compare `on` fst) counts


predict :: Tree -> [Float] -> String
predict (Root attr split left right) record = correctTree `predict` record
    where
        desiredAttr = record !! attr
        correctTree
            | desiredAttr <= split = left
            | otherwise            = right

predict (Leaf value) record = value

numAttrs :: DS -> Int
numAttrs (DS (h: _) _ _) = length h


attrList :: DS -> [Int]
attrList ds = [0..(pred (numAttrs ds))]


unique :: Ord a => [a] -> [a]
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
        DS attrs classes classSet = ds

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

        prob :: Float -> Int -> Float
        prob 0 _ = 0
        prob _ 0 = 0
        prob len count = (fromIntegral count / len)^2

        leftProbs  = map (prob leftLen) leftCounts
        rightProbs = map (prob rightLen) rightCounts

        giniLeft  = 1 - sum leftProbs
        giniRight = 1 - sum rightProbs

        giniTotal = giniLeft * (leftLen / total) + giniRight * (rightLen / total)
    in
        giniTotal

-- data Tree = Root Int Float Tree Tree | Leaf Int Float String String
-- minimizeGini :: DS -> [Int] -> (Int, Float)

createTree :: DS -> [Int] -> Int -> Tree
createTree (DS _ classes _) trainAttrs 1 = Leaf (mode classes)

createTree ds trainAttrs depth =
    let 
        (DS attrs classes uniqueClasses) = ds
        (splitOn, threshold)             = minimizeGini ds trainAttrs

        splitAttr      = map (!! splitOn) attrs
        zipped         = zip attrs classes

        leftMask       = map (<= threshold) splitAttr
        rightMask      = map (>  threshold) splitAttr

        attributesLeft = filter (/= splitOn) trainAttrs

        valuesLeft     = filterByMask attrs leftMask
        classesLeft    = filterByMask classes leftMask
        leftTree       = createTree (DS valuesLeft classesLeft uniqueClasses) attributesLeft (pred depth)

        valuesRight    = filterByMask attrs rightMask
        classesRight   = filterByMask classes rightMask
        rightTree      = createTree (DS valuesRight classesRight uniqueClasses) attributesLeft (pred depth)

        tree           = Root splitOn threshold leftTree rightTree
    in
        tree

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
        nonEmpty      = filter (not . null) allLines
        untyped       = if header then tail nonEmpty else nonEmpty
        split         = map (splitStr separator) untyped
        attrCols      = map init split
        classes       = map last split
        uniqueClasses = unique classes
        attributes    = map (map (if dataType == "char" then ticTacToeToFloat else readAttr)) attrCols
        df            = DS attributes classes uniqueClasses
        attrIndices   = attrList df

    putStrLn "Dataset Loaded"

    let tree = createTree df attrIndices depth

    putStrLn "Shutting down"

