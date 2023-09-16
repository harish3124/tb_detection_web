import React from 'react'

const NavButton = ({name, viewSetter}) => {
    const handleClick = () => {
        viewSetter(name.toLowerCase())
    }
  return (
    <div onClick={handleClick}>{name}</div>
  )
}

export default NavButton
