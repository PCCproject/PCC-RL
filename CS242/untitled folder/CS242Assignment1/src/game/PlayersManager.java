package game;

import game.Player;
import java.util.Random;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;


public class PlayersManager {
	public ArrayList <Player> allPlayers;
	private final int MIN_NUM_PLAYERS = 1;
	private final int MAX_NUM_PLAYERS = 10;
	
	// Order of play. clockwise == TRUE means order of play is from smaller number to larger number. FALSE means otherwise.
	private boolean clockwise;
	
	// the index of the current player. 
	private int currentPlayerId;
	
	
	public PlayersManager() {
		this.allPlayers = new ArrayList<Player>();
		
		// the turn order and starting player is randomly initialized
		Random random = new Random();
		this.clockwise = random.nextBoolean();
		this.currentPlayerId = ThreadLocalRandom.current().nextInt(this.MIN_NUM_PLAYERS-1, this.MAX_NUM_PLAYERS);
	}
	
	/**
	 * Get the number of players.
	 * @return number of players.
	 */
	public int getNumPlayers(){
		return this.allPlayers.size();
	}
	
	/**
	 * Get the current player
	 * @return the current player, or null if the current player's id is out of bound.
	 */
	public Player getCurrentPlayer() {
		Player currentPlayer;
		try {
			currentPlayer = this.allPlayers.get(this.currentPlayerId);
		} catch (Exception e){
			currentPlayer = null;
		}
		return currentPlayer;
	}
	
	
	/**
	 * Get the next player
	 * @return the next player, or null if the next player's id is out of bound.
	 */
	public Player getNextPlayer() {
		Player nextPlayer;
		int nextPlayerId;
		
		if (this.clockwise == true) {
			nextPlayerId = (this.currentPlayerId + 1) % getNumPlayers();
		} else {
			nextPlayerId = (this.currentPlayerId - 1) % getNumPlayers();
		}
		if (nextPlayerId < 0) {
			nextPlayerId += getNumPlayers();
		}
		
		try {
			nextPlayer = this.allPlayers.get(this.currentPlayerId);
		} catch (Exception e){
			nextPlayer = null;
		}
		return nextPlayer;
	}
	
	
	/**
	 * Add a player to the group of players if allowed
	 * @param player to be add.
	 * @return true if it adds a player successfully , false otherwise.
	 */
	public boolean addPlayer(Player player){
		int numOfPlayers = getNumPlayers();
		if (numOfPlayers >= this.MIN_NUM_PLAYERS && numOfPlayers <= this.MAX_NUM_PLAYERS) {
			this.allPlayers.add(player);
			return true;
		}
		else {
			return false;
		}
	}
	
	
	/**
	 * Reverse the order of play.
	 */
	public void reversePlayerDirection() {
		this.clockwise = !this.clockwise;
	}
	
	
	/**
	 * Move the index of the current player in the set direction, so that it's the next player's turn.
	 */
	public void rotatePlayer() {
		int numOfPlayers = getNumPlayers();
		
		if (this.clockwise == true) {
			this.currentPlayerId = (this.currentPlayerId + 1) % numOfPlayers;
		} else {
			this.currentPlayerId = (this.currentPlayerId - 1) % numOfPlayers;
		}
		
		if (this.currentPlayerId < 0) {
			this.currentPlayerId += numOfPlayers;
		}
	}
	
	
	/**
	 * Return the ID of the player whose hand has 0 card (winner), if exists. Return -1 otherwise.
	 * @return the ID of the winning player. If winner does not exist, return -1.
	 */
	public int findWinner(){
		int numOfPlayers = getNumPlayers();
		
		for(int j = 0; j < numOfPlayers; j++){
			if(this.allPlayers.get(j).getNumCards() == 0)
				return j;
		}
		
		return -1;
	}
	
	
}
