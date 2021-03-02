package unocard;
import java.util.*;

public class DrawPile extends CardPile{
	
	// The deck has 108 cards
	public final static int TOTAL_NUM_CARDS = 108;
	
	/*
	 * An initial draw pile has:
	 * Four “Wild” cards (ones look like color wheels)
	 * Four “Wild Draw Four” cards
	 * For each color in red, yellow, green and blue:
	 * One “0” card
	 * Two sets of “1” - “9” cards
	 * Two “Skip” cards
	 * Two “Reverse” cards
	 * Two “Draw Two” cards
	 */
	public DrawPile() {
		
		for(NumberCard.Color color : NumberCard.Color.values()){
			
			this.cardPile.add(new NumberCard(color,0));
			
			for (int i = 1; i < 10; i++) {
				this.cardPile.add(new NumberCard(color,i));
				this.cardPile.add(new NumberCard(color,i));
			}
        }
		
        for(ActionCard.Color color : ActionCard.Color.values()){
            
            this.cardPile.add(new SkipCard(color));
            this.cardPile.add(new ReverseCard(color));
            this.cardPile.add(new DrawTwoCard(color));
            
            this.cardPile.add(new SkipCard(color));
            this.cardPile.add(new ReverseCard(color));
            this.cardPile.add(new DrawTwoCard(color));
        }

        for(int i = 0; i < 4; i++) {
        	this.cardPile.add(new WildCard());
        	this.cardPile.add(new DrawFourWildCard());
        }
			
	}
	
	
	/**
	 * Get and Remove the Uno Card on top of the draw pile.
	 * @return the Uno Card on top. Returns null if index out of bound.
	 */
	public UnoCard getTopCard() {
		int numOfCards = getNumUnoCards();
		return removeCard(numOfCards-1);
	}
	
	
	/**
	 * Shuffle the draw pile.
	 */
    public void shuffle() {
    	Collections.shuffle(this.cardPile);
    }
	
	
}
