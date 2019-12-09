/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file Key.h
 * @brief
 * @author Richard Roberts
 * @author Alex Cunningham
 * @date Feb 20, 2012
 */

#include "semantic_slam/Key.h"
#include "semantic_slam/Symbol.h"

#include <boost/lexical_cast.hpp>
#include <iostream>

using namespace std;

/* ************************************************************************* */
string
_defaultKeyFormatter(Key key)
{
    const Symbol asSymbol(key);
    if (asSymbol.chr() > 0)
        return (string)asSymbol;
    else
        return boost::lexical_cast<string>(key);
}
